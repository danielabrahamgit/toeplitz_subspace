from pathlib import Path
from collections import OrderedDict
from warnings import warn
import time
from typing import Optional, Tuple
import logging

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

from timing import Timer
import toep
from pad import PadLast

logger = logging.getLogger(__name__)

__all__ = ['SubspaceLinopFactory']

def batch_iterator(total, batch_size):
    assert total > 0, f'batch_iterator called with {total} elements'
    delim = list(range(0, total, batch_size)) + [total]
    return zip(delim[:-1], delim[1:])

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
        R: Number of interleaves (deprecated)

        trj: [T D K] all kspace trajectories, in units of rad/voxel ([-pi, pi])
        phi: [A T] temporal subspace basis
        mps: [C H W] coil sensitivities
        sqrt_dcf: [T K] Optional density compensation
        subsamp_idx: [T] Useful when trajectories repeat at multiple timepoints.
            - subsamp_idx[t] = [r], where r is the subsampling index in 0,...,R-1 of that trajectory
            - TODO: support multiple trajectories per TR more easily, i.e. [T N]

        Special cases:
        trj: [T D K] there is one trajectory per time point (i.e. R == T)
            - K can include multiple sub-trajectories
            - sqrt_dcf: [T K] to match. Also, the dcf should be computed on a per-T basis
            - subsamp_idx = torch.arange(T). Self-explanatory
        trj: [1 D K] there is one trajectory for all time points (i.e. R == 1)
            - sqrt_dcf: [1 K]
            - subsamp_idx = torch.zeros(T).

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
            if R == 1:
                warn('Assuming single trajectory per timepoint')
                subsamp_idx = torch.zeros(T).long()
            elif R > 1:
                assert R == T, 'If no subsampling mask provided and number of interleaves > 1, need one interleaf for each timepoint.'
                warn('Assuming trajectories and timepoints correspond')
                subsamp_idx = torch.arange(T)
        else:
            assert subsamp_idx.shape == (T,), 'Subsampling mask must map from time to subsampling index'
        self.subsamp_idx = nn.Parameter(subsamp_idx, requires_grad=False)

        self.nufft = KbNufft(im_size)
        self.nufft_adjoint = KbNufftAdjoint(im_size)

    def get_forward(
            self,
            norm: Optional[str] = 'sigpy',
            coil_batch: Optional[int] = None,
    ):
        R, D = self.trj.shape[:2]
        A = self.ishape[0]
        T, C, K = self.oshape
        scale_factor = 1.
        if norm == 'sigpy':
            norm = 'ortho'
            scale_factor = 2 ** (D/2)

        def A_func_old(x: torch.Tensor):
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

        def A_func(x: torch.Tensor):
            """
            x: [A *im_size]
            """
            assert x.shape == self.ishape, f'Shape mismatch: x: {x.shape}, expected {self.ishape}'
            # Apply subspace basis
            x = torch.einsum('at,a...->t...', self.phi, x[:, None, ...]) # [T C *im_size]
            y = scale_factor * self.nufft(x, self.trj, smaps=self.mps, norm=norm)  # [T C K]
            y = self.sqrt_dcf[:, None, :] * y # [T C K]
            # Subsample
            return y # [T C K]
        if coil_batch is None:
            return A_func, self.ishape, self.oshape

        def A_func(x: torch.Tensor):
            """
            x: [A *im_size]
            """
            assert x.shape == self.ishape, f'Shape mismatch: x: {x.shape}, expected {self.ishape}'
            # Apply subspace basis
            x = torch.einsum('at,a...->t...', self.phi, x[:, None, ...]) # [T C *im_size]
            y = torch.zeros((T, C, K), dtype=torch.complex64, device=x.device)
            for c, d in batch_iterator(C, coil_batch):
                y[:, c:d, :] = scale_factor * self.nufft(x,
                                                         self.trj,
                                                         smaps=self.mps[c:d],
                                                         norm=norm)  # [T C K]
            y = self.sqrt_dcf[:, None, :] * y # [T C K]
            # Subsample
            return y # [T C K]
        return A_func, self.ishape, self.oshape

    def get_adjoint(
            self,
            norm: Optional[str] = 'sigpy',
            coil_batch: Optional[int] = None
    ):
        """
        Note: Don't forget to apply sqrt_dcf to y before applying this
        """
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

        def AH_func_old(y: torch.Tensor):
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

        def AH_func(y: torch.Tensor):
            assert y.shape == self.oshape, f'Shape mismatch: y: {y.shape}, expected {self.oshape}'
            # y_out = torch.zeros((T, C, K), device=y.device).type(y.dtype)
            # Apply adjoint density compensation
            y = self.sqrt_dcf[:, None, :] * y
            # Apply Adjoint NUFFT and coils
            x = scale_factor * self.nufft_adjoint(y, self.trj, smaps=self.mps, norm=norm) # [T H W]
            # Remove leftover coil dim
            x = x[:, 0, ...]
            orig_xshape = x.shape[1:]
            # Apply adjoint subspace
            x = rearrange(x, 't ... -> t (...)')
            x = torch.einsum('at,td->ad', torch.conj(self.phi), x)
            x = x.reshape(x.shape[0], *orig_xshape)
            return x
        if coil_batch is None:
            return AH_func, self.oshape, self.ishape

        def AH_func(y: torch.Tensor):
            assert y.shape == self.oshape, f'Shape mismatch: y: {y.shape}, expected {self.oshape}'
            # y_out = torch.zeros((T, C, K), device=y.device).type(y.dtype)
            # Apply adjoint density compensation
            y = self.sqrt_dcf[:, None, :] * y
            # Apply Adjoint NUFFT and coils
            # Parallelize across coils to save memory
            x = 0
            for c, d in batch_iterator(C, coil_batch):
                x_coil = scale_factor * self.nufft_adjoint(y[:, c:d],
                                                           self.trj,
                                                           smaps=self.mps[c:d], norm=norm) # [T H W]
                x += x_coil[:, 0, ...]
            # Remove leftover coil dim
            #x = x[:, 0, ...]
            orig_xshape = x.shape[1:]
            # Apply adjoint subspace
            x = rearrange(x, 't ... -> t (...)')
            x = torch.einsum('at,td->ad', torch.conj(self.phi), x)
            x = x.reshape(x.shape[0], *orig_xshape)
            return x

        return AH_func, self.oshape, self.ishape

    def get_normal(self, kernels: Optional[torch.Tensor] = None, **kwargs):
        """
        Get the normal operator (i.e. A^H A)

        """
        if kernels is None:
            A_func, _, _ = self.get_forward(**kwargs)
            AH_func, _, _ = self.get_adjoint(**kwargs)
            def AHA_func(x):
                return AH_func(A_func(x))
            return AHA_func, self.ishape, self.ishape
        return self.get_normal_toeplitz(kernels, **kwargs)

    def get_normal_toeplitz(
            self,
            kernels: torch.Tensor,
            batched: bool = True,
            oversamp_factor: int = 2,
    ):
        """
        oversamp_factor: for toeplitz only, the oversamping factor for the PSF
          - default: 2
        batched: Whether or not the toeplitz operator is batched (i.e. accepts [N A *im_size] tensors)
          - default: True
        """
        A, *im_size = self.ishape
        D = len(im_size)
        T, C, K = self.oshape
        R = self.trj.shape[0]
        padder = PadLast(kernels.shape[-D:], im_size)
        coildim = 1 if batched else 0
        def AHA_func(x):
            """
            x: [[N] A H W [D]]
            """
            dim = tuple(range(-D, 0))
            # Apply sensitivies
            x = x.unsqueeze(coildim)
            x = x * self.mps[:, None, ...] # [[N] C A *im_size]
            # Apply pad
            x = padder(x)
            # Apply Toeplitz'd NUFFT normal op
            Fx = fft.fftn(x, dim=dim, norm='ortho')
            Fx = Fx.unsqueeze(coildim + 1)
            x = (oversamp_factor ** D) * fft.ifftn(
                Fx * kernels, dim=dim, norm='ortho'
            ) # [C Aout Ain *2*im_size]
            x = torch.sum(x, dim=(-D-1)) # Sum over Ain

            # Apply adjoint pad
            x = padder.adjoint(x)
            # Apply adjoint sensitivities
            x = x * torch.conj(self.mps[:, None, ...])
            x = torch.sum(x, dim=(-D-2)) # Sum over coils
            return x
        return AHA_func, self.ishape, self.ishape

    def get_kernels(self, im_size):
        """Simple way of getting kernels with good defaults
        """
        with Timer('compute_weights'):
            weights = toep.compute_weights(
                self.subsamp_idx,
                self.phi,
                self.sqrt_dcf,
                memory_efficient=True,
            )

        with Timer('compute_kernels'):
            kernels = toep.compute_kernels(
                self.trj,
                weights,
                im_size,
                oversamp_factor=2,
            )
        return kernels

    # def get_kernels_cache(
    #         self,
    #         cache_file: Path,
    #         im_size: Tuple,
    #         force_reload: bool = False,
    # ):
    #     if cache_file.is_file() and not force_reload:
    #         kernels = np.load(cache_file)
    #         kernels = torch.from_numpy(kernels)
    #     else:
    #         kernels = self.get_kernels(
    #             im_size,
    #             oversamp_factor=2,
    #             device='cpu',
    #             verbose=True,
    #         )
    #         np.save(cache_file, kernels.detach().cpu().numpy())
    #     return kernels
