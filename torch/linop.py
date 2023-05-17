from collections import OrderedDict
import time
from typing import Optional

from einops import rearrange, repeat
import numpy as np
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
from tqdm import tqdm

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
        T, C, K = self.oshape
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
            verbose=False,
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

        A, *im_size = self.ishape
        T, C, K = self.oshape
        R = self.trj.shape[0]
        # Compute toeplitz embeddings
        if kernels is None:
            if verbose:
                print('> Computing weights...')
            start = time.perf_counter()
            weights = torch.zeros((R, A, A, K), device=device).type(torch.complex64)
            subsamp_mask = torch.zeros((R, T), device=device).type(torch.complex64)
            subsamp_mask[self.subsamp_idx, torch.arange(T)] = 1.
            for a_in in range(A):
                weight = torch.zeros((R, A, K), device=device).type(torch.complex64)
                weight[:, a_in, :] = 1.
                weight = torch.einsum('at,rak->rtk', self.phi, weight)
                weight = weight * subsamp_mask[..., None] * (self.sqrt_dcf[:, None, :] ** 2)
                weight = torch.einsum('at,rtk->rak', torch.conj(self.phi), weight)
                weights[:, :, a_in, :] = weight
            total = time.perf_counter() - start
            if verbose:
                print(f'>> Time: {total} s')


            if verbose:
                print('> Generating kernels...')
            start = time.perf_counter()
            kernel_size = (oversamp_factor * d for d in im_size)
            kernels = torch.zeros((A, A, *kernel_size), device=device).type(torch.complex64)
            for a_in in range(A):
                for a_out in range(A):
                    if a_in > a_out:
                        kernels[a_out, a_in, ...] = kernels[a_in, a_out]
                        continue
                    if verbose:
                        print(f'>> Calculating kernel({a_out}, {a_in})')
                    kernel = calc_toeplitz_kernel(
                        self.trj,
                        im_size,
                        weights=weights[:, None, a_out, a_in, :]  # [I 1 K]
                    )
                    kernels[a_out, a_in, ...] = kernel.sum(dim=0)
            total = time.perf_counter() - start
            if verbose:
                print(f'>> Time: {total} s')

        # Deal with cropping
        def crop(img, pads):
            return img[..., pads]
        D = len(im_size)
        kernel_size = kernel.shape[-D:]
        pads = sum(([(ksz - isz) // 2]*2
                   for ksz, isz in zip(kernel_size, im_size)), start=[]).reverse()  # F.pad requires reversal of dims
        slc = [slice(p[2*i], -p[2*i+1]) for i in range(len(pads)//2)].reverse()
        print(pads)
        print(slc)
        pads.extend([0, 0, 0, 0])   # coil and subspace coefficient dims don't get padded
        def AHA_func(x):
            """
            x: [A H W]
            """
            dim = tuple(range(-D, 0))
            # Apply sensitivies
            x = x[None, :, ...] * self.mps[:, None, ...] # [C A *im_size]
            x = F.pad(x, pads)
            # Apply Toeplitz'd NUFFT normal op
            Fx = fft.fftn(x, dim=dim, norm='ortho')
            # Note that kernel has DC in top left (i.e. not fftshifted)
            # batch_kernel = kernel/torch.sum(torch.abs(kernel)**2, dim=(-2, -1), keepdim=True)
            # batch_kernel = torch.real(kernel)
            # batch_kernel = batch_kernel[:, None, :, :] # I 1 H W
            # Fx: [C Ain 2H 2W]
            # kernels: [Aout Ain 2H 2W]
            x = (oversamp_factor ** D) * fft.ifftn(
                Fx[:, None, ...] * kernels, dim=dim, norm='ortho'
            ) # [C Aout Ain *2*im_size]

            x = torch.sum(x, dim=2) # Sum over Ain
            x = crop(x, slc)
            # Apply adjoint sensitivities
            x = x * torch.conj(self.mps[:, None, ...])
            x = torch.sum(x, dim=0) # Sum over coils
            return x
        return AHA_func, self.ishape, self.ishape
