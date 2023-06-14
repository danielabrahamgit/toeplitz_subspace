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

from .timing import tictoc
from . import toep
from .pad import PadLast

__all__ = ['SubspaceLinopFactory']

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
        R: Number of interleaves

        trj: [R D K] all kspace trajectories, in units of rad/voxel ([-pi, pi])
        phi: [A T] temporal subspace basis
        mps: [C H W] coil sensitivities
        sqrt_dcf: [R K] Optional density compensation
        subsamp_idx: [T] Useful when trajectories repeat at multiple timepoints.
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

        sqrt_dcf = sqrt_dcf if sqrt_dcf is not None else torch.ones((R, K)).type(torch.float32)
        self.sqrt_dcf = nn.Parameter(sqrt_dcf, requires_grad=False)

        if subsamp_idx is None:
            assert R == T, 'If no subsampling mask provided, need one trajectory for each timepoint.'
            subsamp_idx = torch.arange(T)
        else:
            assert subsamp_idx.shape == (T,), 'Subsampling mask must map from time to subsampling index'
        self.subsamp_idx = nn.Parameter(subsamp_idx, requires_grad=False)

        self.nufft = KbNufft(im_size)
        self.nufft_adjoint = KbNufftAdjoint(im_size)

    def get_forward(self, norm: Optional[str] = 'sigpy'):
        R, D = self.trj.shape[:2]
        A = self.ishape[0]
        T, C, K = self.oshape
        scale_factor = 1.
        if norm == 'sigpy':
            norm = 'ortho'
            scale_factor = 2 ** (D/2)

        def A_func(x: torch.Tensor):
            """
            x: [A *im_size]
            """
            assert x.shape == self.ishape, f'Shape mismatch: x: {x.shape}, expected {self.ishape}'
            y = []
            for a in range(A):
                x_a = x[a:a+1, ...]
                x_a = x_a.repeat(R, 1, *(D*(1,)))  # [R 1 *im_size]
                y_a = scale_factor * self.nufft(x_a, self.trj, smaps=self.mps, norm=norm)  # [R C K]
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

    def get_adjoint(self, norm: Optional[str] = 'sigpy'):
        R = self.trj.shape[0]
        A, *im_size = self.ishape
        D = len(im_size)
        T, C, K = self.oshape
        scale_factor = 1/np.prod(im_size)
        if norm == 'ortho':
            scale_factor = 1.
        elif norm == 'sigpy':
            norm = 'ortho'
            scale_factor = (2 ** D/2)

        def AH_func(y: torch.Tensor):
            assert y.shape == self.oshape, f'Shape mismatch: y: {y.shape}, expected {self.oshape}'
            y_out = torch.zeros((R, T, C, K), device=y.device).type(y.dtype)
            # Expand along subsampling dimension
            y_out[self.subsamp_idx, torch.arange(T), :, :] = y
            # Apply adjoint density compensation
            y = self.sqrt_dcf[:, None, None, :] * y_out
            # Apply adjoint subspace
            y = torch.einsum('at,rtck->arck', torch.conj(self.phi), y)
            x = []
            for a in range(A):
                y_a = y[a, ...] # [R C K]
                x_a = scale_factor * self.nufft_adjoint(y_a, self.trj, smaps=self.mps, norm=norm) # [R 1 H W]
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
        """
        if not toeplitz:
            A_func, _, _ = self.get_forward()
            AH_func, _, _ = self.get_adjoint()
            def AHA_func(x):
                return AH_func(A_func(x))
            return AHA_func, self.ishape, self.ishape

        A, *im_size = self.ishape
        D = len(im_size)
        T, C, K = self.oshape
        R = self.trj.shape[0]
        # Compute toeplitz embeddings
        if kernels is None:
            kernels = self.get_kernels(im_size, oversamp_factor, device, verbose)
        padder = PadLast(kernels.shape[-D:], im_size)

        def AHA_func(x):
            """
            x: [A H W]
            """
            dim = tuple(range(-D, 0))
            # Apply sensitivies
            x = x[None, :, ...] * self.mps[:, None, ...] # [C A *im_size]
            # Apply pad
            x = padder(x)
            # Apply Toeplitz'd NUFFT normal op
            Fx = fft.fftn(x, dim=dim, norm='ortho')
            x = (oversamp_factor ** D) * fft.ifftn(
                Fx[:, None, ...] * kernels, dim=dim, norm='ortho'
            ) # [C Aout Ain *2*im_size]
            x = torch.sum(x, dim=2) # Sum over Ain

            # Apply adjoint pad
            x = padder.adjoint(x)
            # Apply adjoint sensitivities
            x = x * torch.conj(self.mps[:, None, ...])
            x = torch.sum(x, dim=0) # Sum over coils
            return x
        return AHA_func, self.ishape, self.ishape

    def get_kernels(self, im_size, oversamp_factor, device, verbose):
        with tictoc('compute_weights', verbose):
            weights = toep.compute_weights(
                self.subsamp_idx,
                self.phi,
                self.sqrt_dcf,
                device=device,
            )

        with tictoc('compute_kernels', verbose):
            kernels = toep.compute_kernels(
                self.trj,
                weights,
                im_size,
                oversamp_factor,
                device,
                verbose,
            )
        return kernels
