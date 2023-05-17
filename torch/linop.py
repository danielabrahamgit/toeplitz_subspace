from collections import OrderedDict
import time
from typing import Optional

from einops import rearrange, repeat
import numpy as np
import sigpy as sp
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from torchkbnufft import (
    KbNufft,
    KbNufftAdjoint,
    ToepNufft,
    calc_toeplitz_kernel,
)

class SubspaceLinopFactory(nn.Module):
    def __init__(
            self,
            trj: torch.Tensor,
            phi: torch.Tensor,
            mps: torch.Tensor,
            sqrt_dcf: Optional[torch.Tensor] = None,
            subsamp_idx: Optional[torch.Tensor] = None,
    ):
        """
        Dimensions:
        T: number of time points
        C: number of coils
        D: dimension of reconstruction (e.g 2D, 3D, etc)
        A: temporal subspace dimension
        K: number of kspace trajectory points

        trj: [R D K] all kspace trajectories, in units of rad/voxel ([-pi, pi])
        phi: [A T] temporal subspace basis
        mps: [C H W] SENSE maps
        sqrt_dcf: [R K] Optional density compensation
        subsamp_idx: [T 1] Useful when trajectories repeat at multiple timepoints.
            - subsamp_idx[t] = [r], where r is the subsampling index in 0,...,R-1 of that trajectory
            - TODO: support multiple trajectories per TR more easily, i.e. [T N]

        """
        super().__init__()
        self.trj = nn.Parameter(trj, requires_grad=False)
        self.phi = nn.Parameter(phi, requires_grad=False)
        self.mps = nn.Parameter(mps, requires_grad=False)

        A, T = self.phi.shape
        C, *im_size = self.mps.shape
        R, D, K = self.trj.shape
        assert D == len(im_size), f'Dimension mismatch: coils have dimension {len(im_size)}, trj has dimension {D}'

        self.ishape = (A, *im_size)
        self.oshape = (T, C, K)

        sqrt_dcf = sqrt_dcf if sqrt_dcf is not None else torch.ones((R, K))
        self.sqrt_dcf = nn.Parameter(sqrt_dcf, requires_grad=False)

        if subsamp_idx is None:
            assert R == T, 'If no subsampling mask provided, need one trajectory for each timepoint.'
            subsamp_idx = torch.arange(T)
        else:
            assert subsamp_idx.shape == (T,), 'Subsampling mask must map from time to subsampling index'
        self.subsamp_idx = nn.Parameter(subsamp_idx, requires_grad=False)

        self.nufft = KbNufft(im_size)
        self.nufft_adjoint = KbNufftAdjoint(im_size)

    def get_forward(self):
        R, D = self.trj.shape[:2]
        A = self.ishape[0]
        T, C, K = self.oshape
        def A_func(x: torch.Tensor):
            """
            x: [A *im_size]
            """
            assert x.shape == self.ishape, f'Shape mismatch: x: {x.shape}, expected {self.ishape}'
            y = []
            for a in range(A):
                x_a = x[a:a+1, ...]
                x_a = x_a.repeat(R, 1, *(D*(1,)))  # [R 1 *im_size]
                y_a = self.nufft(x_a, self.trj, smaps=self.mps)  # [R C K]
                y.append(y_a)
            y = torch.stack(y)  # [A R C K]
            # Apply subspace basis
            y = torch.einsum('at,arck->rtck', self.phi, y)
            # Apply DCF (TODO: might change if multiple trajectories per timepoint)
            y = self.sqrt_dcf[:, None, None, :] * y # [R T C K]
            # Subsample
            y = y[self.subsamp_idx, torch.arange(T), ...]
            return y # [T C K]
        return A_func, self.ishape, self.oshape

    def get_adjoint(self):
        A, H, W = self.ishape
        C, I, T, K = self.oshape
        def AH_func(y: torch.Tensor):
            assert y.shape == self.oshape, f'Shape mismatch: y: {y.shape}, expected {self.oshape}'
            y_out = torch.zeros((I, T, C, K), device=y.device).type(y.dtype)
            # Expand along interleave dimension
            y_out[self.subsamp_idx, torch.arange(T), :, :] = y
            # Apply adjoint density compensation
            y = self.sqrt_dcf[..., None, :] * y_out
            # Apply adjoint subspace
            y = torch.einsum('at,itck->aick', torch.conj(self.phi), y)
            x = []
            for a in range(A):
                y_a = y[a, ...] # [I C K]
                x_a = 1/(H*W) * self.nufft_adjoint(y_a, self.trj, smaps=self.mps) # [I 1 H W]
                x_a = x_a.sum(0) # [1 H W]
                x.append(x_a)
            x = torch.stack(x) # [A 1 H W]
            return x[:, 0, ...]
        return AH_func, self.oshape, self.ishape

    def get_normal(
            self,
            toeplitz: bool = False,
            oversamp_factor: int = 2,
            device='cpu',
            kernels: Optional[torch.Tensor] = None,
    ):
        """
        Get the normal operator (i.e. A^H A)

        oversamp_factor: for toeplitz only, the oversamping
        factor for the PSF

        Optional argument to use the toeplitz embedding,
        which is memory intensive but much faster.
        C: number of coils
        A: number of subspace coefficents
        H, W: shape of image
        I: Number of interleaves
        T: Number of TRs (timepoints)
        K: Number of kspace trajectory points
        """
        if not toeplitz:
            A_func, _, _ = self.get_forward()
            AH_func, _, _ = self.get_adjoint()
            def AHA_func(x):
                return AH_func(A_func(x))
            return AHA_func, self.ishape, self.ishape

        A, H, W = self.ishape
        C, I, T, K = self.oshape
        im_size = (H, W)
        # Compute toeplitz embeddings
        if kernels is None:
            print('> Computing weights...')
            start = time.perf_counter()
            weights = torch.zeros((I, A, A, K), device=device).type(torch.complex64)
            for a_in in range(A):
                weight = torch.zeros((I, A, K), device=device).type(torch.complex64)
                weight[:, a_in, :] = 1.
                weight = torch.einsum('at,iak->itk', self.phi, weight)
                weight = weight * self.subsamp_mask * (self.sqrt_dcf ** 2)
                weight = torch.einsum('at,itk->iak', torch.conj(self.phi), weight)
                weights[:, :, a_in, :] = weight
            total = time.perf_counter() - start
            print(f'>> Time: {total} s')

            print('> Generating kernels...')
            start = time.perf_counter()
            kernel_size = (oversamp_factor * d for d in im_size)
            kernels = torch.zeros((A, A, *kernel_size), device=device).type(torch.complex64)
            for a_in in range(A):
                for a_out in range(A):
                    kernel = calc_toeplitz_kernel(
                        self.trj,
                        im_size,
                        weights=weights[:, None, a_out, a_in, :]  # [I 1 K]
                    )
                    kernels[a_out, a_in, ...] = kernel.sum(dim=0)
            total = time.perf_counter() - start
            print(f'>> Time: {total} s')

        def crop(img, pad_height, pad_width):
            return img[..., pad_height:-pad_height, pad_width:-pad_width]
        kernel_size = kernel.shape[-2:]
        pad_height = (kernel_size[0] - im_size[0]) // 2
        pad_width = (kernel_size[1] - im_size[1]) // 2
        pad = (
            pad_width, pad_width,
            pad_height, pad_height,
            0, 0,
            0, 0,
        )
        def AHA_func(x):
            """
            x: [A H W]
            """
            # Apply sensitivies
            x = x[None, :, ...] * self.mps[:, None, ...] # [C A H W]
            x = F.pad(x, pad)
            # Apply Toeplitz'd NUFFT normal op
            oversamp_h = kernel_size[0]/im_size[0]
            oversamp_w = kernel_size[1]/im_size[1]
            Fx = fft.fftn(x, dim=(-2, -1), norm='ortho')
            # Note that kernel has DC in top left (i.e. not fftshifted)
            # batch_kernel = kernel/torch.sum(torch.abs(kernel)**2, dim=(-2, -1), keepdim=True)
            # batch_kernel = torch.real(kernel)
            # batch_kernel = batch_kernel[:, None, :, :] # I 1 H W
            # Fx: [C Ain 2H 2W]
            # kernels: [Aout Ain 2H 2W]
            x = oversamp_h * oversamp_w * fft.ifftn(
                Fx[:, None, ...] * kernels, dim=(-2, -1), norm='ortho'
            ) # [C Aout Ain 2H 2W]

            x = torch.sum(x, dim=2) # Sum over Ain
            x = crop(x, pad_height, pad_width)
            # Apply adjoint sensitivities
            x = x * torch.conj(self.mps[:, None, ...])
            x = torch.sum(x, dim=0) # Sum over coils
            return x
        return AHA_func, self.ishape, self.ishape

    # def get_ops(self, toeplitz: bool = False):
    #     A, input_shape, output_shape = self._forward()
    #     AH, _, _ = self._adjoint()
    #     AHA, _, _ = self._normal(toeplitz)

    #     class MRFLinop(torch.autograd.Function):
    #         ishape: Tuple = input_shape
    #         oshape: Tuple = output_shape

    #         @staticmethod
    #         def forward(ctx, x):
    #             return A(x)

    #         @staticmethod
    #         def backward(ctx, x_grad):
    #             return AH(x_grad)

    #     class MRFLinopAdjoint(torch.autograd.Function):
    #         ishape: Tuple = output_shape
    #         oshape: Tuple = input_shape

    #         @staticmethod
    #         def forward(ctx, x):
    #             return AH(x)

    #         @staticmethod
    #         def backward(ctx, x_grad):
    #             return A(x_grad)

    #     return MRFLinop, MRFLinopAdjoint
