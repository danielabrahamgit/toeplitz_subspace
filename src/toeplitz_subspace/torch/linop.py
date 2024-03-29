from pathlib import Path
from collections import OrderedDict
from warnings import warn
import time
from typing import Optional, Tuple, Union
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

from . import toep
from .pad import PadLast

logger = logging.getLogger(__name__)

__all__ = ['SubspaceLinopFactory']

def batch_iterator(total, batch_size):
    assert total > 0, f'batch_iterator called with {total} elements'
    delim = list(range(0, total, batch_size)) + [total]
    return zip(delim[:-1], delim[1:])

def ceildiv(num, denom):
    return -(num//-denom)

def batch_tqdm(total, batch_size, **tqdm_kwargs):
    return tqdm(
        batch_iterator(total, batch_size),
        total=ceildiv(total, batch_size),
        **tqdm_kwargs
    )

class SubspaceLinopFactory(nn.Module):
    def __init__(
            self,
            trj: torch.Tensor,
            phi: torch.Tensor,
            mps: torch.Tensor,
            sqrt_dcf: Optional[torch.Tensor] = None,
            subsamp_idx: Optional[torch.Tensor] = None,
            oversamp_factor: float = 2.,
    ):
        """
        Dimensions:
        T: number of time points
        C: number of coils
        D: dimension of reconstruction (e.g 2D, 3D, etc)
        A: temporal subspace dimension
        K: number of kspace trajectory points
        R: total number of interleaves (may be different than T if trajectories repeat)
        I: number of interleaves per timepoint

        trj: [R D K] all kspace trajectories, in units of rad/voxel ([-pi, pi])
        sqrt_dcf: [R K] Optional density compensation
        phi: [A T] temporal subspace basis
        mps: [C H W] coil sensitivities
        subsamp_idx: [I, T] Useful when trajectories repeat at multiple timepoints.
            - subsamp_idx[:, t] = [j0, j1, ..., j(I-1)] is of length I, where the j's index into [0, R-1]

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
        self.A, self.T = phi.shape
        self.C, *self.im_size = mps.shape
        self.R, self.D, self.K = trj.shape
        assert self.D == len(self.im_size), f'Dimension mismatch: coils have dimension {len(self.im_size)}, trj has dimension {self.D}'

        self.trj = nn.Parameter(trj, requires_grad=False)
        self.phi = nn.Parameter(phi, requires_grad=False)
        self.mps = nn.Parameter(mps, requires_grad=False)
        sqrt_dcf = sqrt_dcf if sqrt_dcf is not None else torch.ones((self.R, self.K)).type(torch.float32)
        self.sqrt_dcf = nn.Parameter(sqrt_dcf, requires_grad=False)

        if subsamp_idx is not None:
            assert subsamp_idx.shape[1] == self.T, f'subsamp_idx has first dimension {subsamp_idx.shape[0]}, expected {self.T}'
            self.I, _ = subsamp_idx.shape
        else:
            logger.warning('No subsamp_idx provided')
            if self.R == 1:
                logger.warning('Assuming one trajectory per timepoint')
                subsamp_idx = torch.zeros((1, self.T)).long()
            elif self.R > 1:
                assert self.R == self.T, 'If no subsampling mask provided and number of interleaves > 1, need one interleaf for each timepoint.'
                logger.warning('Assuming trajectories and timepoints are in 1:1 correspondence')
                subsamp_idx = torch.arange(self.T)[None, :]
            else:
                raise ValueError(f'No subsamp_idx provided and subsampling is ambiguous for trj: {trj.shape} and {T} timepoints.')
            self.I = 1
        self.subsamp_idx = nn.Parameter(subsamp_idx, requires_grad=False)

        # NUFFTs
        self.oversamp_factor = oversamp_factor
        self.nufft = KbNufft(
            self.im_size,
            grid_size=tuple(int(oversamp_factor*d) for d in self.im_size),
        )
        self.nufft_adjoint = KbNufftAdjoint(
            self.im_size,
            grid_size=tuple(int(oversamp_factor*d) for d in self.im_size),
        )

        # Input and output shapes
        self.ishape = (self.A, *self.im_size)
        self.oshape = (self.I, self.T, self.C, self.K)

    def get_forward(
            self,
            norm: Optional[str] = 'sigpy',
            coil_batch: Optional[int] = None,
            trj_batch: Optional[int] = None,
    ):
        """
        norm: Recommended 'sigpy' - scales tkbn's nufft to be equivalent to sigpy's nufft
        """
        I, T, C, K, A, R, D = self.I, self.T, self.C, self.K, self.A, self.R, self.D
        scale_factor = 1.
        coil_batch = coil_batch if coil_batch is not None else C
        trj_batch = trj_batch if trj_batch is not None else T
        if norm == 'sigpy':
            norm = 'ortho'
            scale_factor = self.oversamp_factor
        def A_func(x: torch.Tensor):
            """
            x: [A *im_size]
            """
            assert x.shape == self.ishape, f'Shape mismatch: x: {x.shape}, expected {self.ishape}'
            y = torch.zeros((I, T, C, K), device=x.device, dtype=torch.complex64)
            for c1, c2 in tqdm(batch_iterator(C, coil_batch),
                             total=C//coil_batch,
                             desc='Forward (C)',
                             leave=False):
                for t1, t2 in tqdm(batch_iterator(T, trj_batch),
                                   total=R//trj_batch,
                                   desc='Forward (R)',
                                   leave=False):
                # for a1, a2 in tqdm(batch_iterator(A, sub_batch),
                #                    total=A//sub_batch,
                #                    desc='Forward (A)',
                #                    leave=False):
                    for a in range(A):
                        x_a = x[a]
                        x_a = x_a.repeat(t2-t1, 1, *(D*(1,)))  # [R 1 *im_size]
                        trj = self.trj[self.subsamp_idx][:, t1:t2, ...]
                        trj = rearrange(trj, 'i t d k -> t d (i k)')
                        y_a = scale_factor * self.nufft(
                            x_a,
                            trj,
                            smaps=self.mps[c1:c2],
                            norm=norm,
                        )  # [R C K]
                        y_a = rearrange(y_a, 't c (i k) -> i t c k', i=I)
                        y_a = y_a * self.sqrt_dcf[self.subsamp_idx][:, t1:t2, None, :]
                        y_a = y_a * self.phi[a, t1:t2, None, None]
                        y[:, t1:t2, c1:c2, :] += y_a
            return y
        return A_func, self.ishape, self.oshape

    def get_adjoint(
            self,
            norm: Optional[str] = 'sigpy',
            coil_batch: Optional[int] = None,
            trj_batch: Optional[int] = None,
            nufft_device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Note: Don't forget to apply sqrt_dcf to y before applying this
        coil_batch: How many coils to compute inufft over at once
        trj_batch: How many trajectories to compute inufft over at once
        nufft_device: which device to use for nufft
        """
        I, T, C, K, A, R, D = self.I, self.T, self.C, self.K, self.A, self.R, self.D
        scale_factor = 1/np.prod(self.im_size)
        coil_batch = coil_batch if coil_batch is not None else C
        trj_batch = trj_batch if trj_batch is not None else T
        if norm == 'ortho':
            scale_factor = 1.
        elif norm == 'sigpy':
            norm = 'ortho'
            scale_factor = self.oversamp_factor

        def AH_func(y: torch.Tensor):
            assert y.shape == self.oshape, f'Shape mismatch: y: {y.shape}, expected {self.oshape}'
            x = torch.zeros((A, *self.im_size), device=y.device, dtype=torch.complex64)
            for c, d in tqdm(batch_iterator(C, coil_batch),
                             total=C//coil_batch,
                             desc='Adjoint (C)',
                             leave=False):
                for e, f in tqdm(batch_iterator(T, trj_batch),
                                 total=T//trj_batch,
                                 desc='Adjoint (T)',
                                 leave=False):
                    for a in range(A):
                        y_a = y[:, e:f, c:d, :] * torch.conj(self.phi)[a, e:f, None, None] # [I Tsub, D K]
                        sqrt_dcf = self.sqrt_dcf[self.subsamp_idx[:, e:f], None, :] # [I Tsub 1 K]
                        y_a = y_a * sqrt_dcf
                        flat_idx = rearrange(self.subsamp_idx[:, e:f], 'i t -> (i t)')
                        flat_y_a = rearrange(y_a, 'i t c k -> (i t) c k')
                        y_a = torch.zeros(
                            (R, d-c, K), device=y.device, dtype=torch.complex64
                        ).index_add_(0, flat_idx, flat_y_a)
                        y_a = y_a[e:f]
                        x_a = scale_factor * self.nufft_adjoint(
                            y_a,
                            self.trj[e:f],
                            smaps=self.mps[c:d],
                            norm=norm,
                        ) # [R 1 H W], the 1 is the reduced coil dimension
                        x[a, ...] += torch.sum(x_a[:, 0, ...], dim=0)
            return x

        #if coil_batch is None and trj_batch is None:
        return AH_func, self.oshape, self.ishape


    def get_normal(self, kernels: Optional[torch.Tensor] = None,
                   **batch_kwargs):
        """
        Get the normal operator (i.e. A^H A)

        """
        if kernels is None:
            A_func, _, _ = self.get_forward(**batch_kwargs)
            AH_func, _, _ = self.get_adjoint(**batch_kwargs)
            def AHA_func(x):
                return AH_func(A_func(x))
            return AHA_func, self.ishape, self.ishape
        return self.get_normal_toeplitz(kernels, **kwargs)

    def get_normal_toeplitz(
            self,
            kernels: torch.Tensor,
            norm: str = 'sigpy',
            batched_input: bool = False,
            coil_batch: Optional[int] = None,
            sub_batch: Optional[int] = None,
            nufft_device: Optional[torch.device] = None,
    ):
        """
        oversamp_factor: for toeplitz only, the oversamping factor for the PSF
          - default: 2
        batched: Whether or not the toeplitz operator is batched (i.e. accepts [N A *im_size] tensors)
          Mostly useful for unrolled networks
          - default: False
        """
        I, T, C, K, A, R, D = self.I, self.T, self.C, self.K, self.A, self.R, self.D
        padder = PadLast(kernels.shape[-D:], self.im_size)

        scale_factor = 1.
        if norm == 'sigpy':
            # NOT dimension-specific, the square is once for the forward, once for adjoint.
            scale_factor = self.oversamp_factor ** 2


        coildim = 1 if batched_input else 0
        coil_batch = coil_batch if coil_batch is not None else 1
        sub_batch = sub_batch if sub_batch is not None else 1
        batch_slice = [slice(None)] if batched_input else [] # allows slicing of coils

        def AHA_func(x: torch.Tensor):
            """
            x: [[N] A H W [D]]
            """
            device = x.device
            out = torch.zeros_like(x)
            dim = tuple(range(-D, 0))
            # Apply sensitivies
            x = x.unsqueeze(coildim)
            x = x * self.mps[:, None, ...] # [[N] C Ain *im_size]
            for c1, c2 in batch_tqdm(C, coil_batch, desc='Normal (C)', leave=False):
                for a_in in range(A):
                    in_slc = batch_slice + [slice(c1, c2), a_in]
                    x_a_in = x[in_slc] # [[N] C *im_size
                    x_a_in = padder(x_a_in)
                    Fx_a_in = fft.fftn(x_a_in, dim=dim, norm='ortho')
                    kernels_dev = kernels[:, a_in].to(device)
                    for a_out in range(A):
                        kernel = kernels_dev[a_out]
                        x_a_out = fft.ifftn(Fx_a_in * kernel, dim=dim, norm='ortho')
                        x_a_out = padder.adjoint(x_a_out)
                        # apply adjoint coil
                        x_a_out = torch.sum(x_a_out * torch.conj(self.mps[c1:c2]),
                                            dim=coildim)
                        out_slc = batch_slice + [a_out]
                        out[out_slc] += scale_factor * x_a_out
            return out


        return AHA_func, self.ishape, self.ishape

    def get_kernels(
            self,
            im_size,
            **batch_kwargs,
    ):
        """Compute kernels with the current set of data
        batch_kwargs: misc kwargs to pass to toep module
        """
        A, T = self.phi.shape
        device = self.phi.device
        dtype = torch.complex64
        kernel_size = tuple(int(self.oversamp_factor*d) for d in im_size)
        kernels = torch.zeros((A, A, *kernel_size), dtype=dtype, device='cpu')

        kernels = toep._compute_weights_and_kernels(
            im_size,
            self.trj[self.subsamp_idx, ...],
            self.phi,
            self.sqrt_dcf[self.subsamp_idx, ...],
            self.oversamp_factor,
            kernels,
            apply_scaling=False,
            **batch_kwargs,
        )
        D = len(im_size)
        #scale_factor = 1/(np.prod(im_size) * self.oversamp_factor)
        scale_factor = 1/(np.prod(im_size) * self.oversamp_factor ** D)
        return kernels * scale_factor
