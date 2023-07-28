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

from .timing import Timer
from . import toep
from .pad import PadLast

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
        assert self.D == len(self.im_size), f'Dimension mismatch: coils have dimension {len(im_size)}, trj has dimension {D}'

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
            self
,
            norm: Optional[str] = 'sigpy',
            coil_batch: Optional[int] = None,
    ):
        I, T, C, K, A, R, D = self.I, self.T, self.C, self.K, self.A, self.R, self.D
        scale_factor = 1.
        if coil_batch is None:
            coil_batch = C
        if norm == 'sigpy':
            norm = 'ortho'
            scale_factor = self.oversamp_factor
        def A_func(x: torch.Tensor):
            """
            x: [A *im_size]
            """
            assert x.shape == self.ishape, f'Shape mismatch: x: {x.shape}, expected {self.ishape}'
            y = torch.zeros((I, T, C, K), device=x.device, dtype=torch.complex64)
            for c, d in tqdm(batch_iterator(C, coil_batch),
                             total=C//coil_batch,
                             desc='A',
                             leave=False):
                for a in range(A):
                    x_a = x[a:a+1, ...]
                    x_a = x_a.repeat(R, 1, *(D*(1,)))  # [R 1 *im_size]
                    y_a = scale_factor * self.nufft(
                        x_a,
                        self.trj,
                        smaps=self.mps[c:d],
                        norm=norm,
                    )  # [R C K]
                    y_a *= self.sqrt_dcf[:, None, :]
                    y_a = y_a[self.subsamp_idx, ...] # [I T C K]
                    y_a *= self.phi[a, :, None, None]
                    y[..., c:d, :] += y_a
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
                             desc='AH',
                             leave=False):
                for e, f in tqdm(batch_iterator(T, trj_batch)):
                    for a in range(A):
                        y_a = y[:, e:f, :, :] * torch.conj(self.phi)[a, e:f, None, None] # [I Tsub, D K]
                        sqrt_dcf = self.sqrt_dcf[self.subsamp_idx[:, e:f], None, :] # [I Tsub 1 K]
                        y_a *= sqrt_dcf
                        flat_idx = rearrange(self.subsamp_idx, 'i t -> (i t)')
                        flat_y_a = rearrange(y_a, 'i t c k -> (i t) c k')
                        y_a = torch.zeros(
                            (R, C, K), device=y.device, dtype=torch.complex64
                        ).index_add_(0, flat_idx, flat_y_a)
                        x_a = scale_factor * self.nufft_adjoint(
                            y_a,
                            self.trj,
                            smaps=self.mps[c:d],
                            norm=norm,
                        ) # [R 1 H W], the 1 is the reduced coil dimension
                        x[a, ...] += torch.sum(x_a[:, 0, ...], dim=0)
            return x

        #if coil_batch is None and trj_batch is None:
        return AH_func, self.oshape, self.ishape


    def get_normal(self, kernels: Optional[torch.Tensor] = None):
        """
        Get the normal operator (i.e. A^H A)

        """
        if kernels is None:
            A_func, _, _ = self.get_forward()
            AH_func, _, _ = self.get_adjoint()
            def AHA_func(x):
                return AH_func(A_func(x))
            return AHA_func, self.ishape, self.ishape
        return self.get_normal_toeplitz(kernels, **kwargs)

    def get_normal_toeplitz(
            self,
            kernels: torch.Tensor,
            batched_input: bool = True,
            coil_batch: Optional[int] = None,
            sub_batch: Optional[int] = None,
            nufft_device: Optional[torch.device] = None,
    ):
        """
        oversamp_factor: for toeplitz only, the oversamping factor for the PSF
          - default: 2
        batched: Whether or not the toeplitz operator is batched (i.e. accepts [N A *im_size] tensors)
          - default: True
        """
        I, T, C, K, A, R, D = self.I, self.T, self.C, self.K, self.A, self.R, self.D
        padder = PadLast(kernels.shape[-D:], self.im_size)

        coildim = 1 if batched_input else 0
        coil_batch = coil_batch if coil_batch is not None else 1
        sub_batch = sub_batch if sub_batch is not None else 1
        batch_slice = [slice(None)] if batched_input else [] # allows slicing of coils
        def AHA_func_old(x):
            """
            x: [[N] A H W [D]]
            """
            if nufft_device is None:
                fft_device = x.device
            else:
                fft_device = nufft_device
            dim = tuple(range(-D, 0))
            # Apply sensitivies
            x = x.unsqueeze(coildim)
            x = x * self.mps[:, None, ...] # [[N] C A *im_size]

            # Sum over coils and Ain (subspace coeffs)
            out = 0.
            for v, w in tqdm(batch_iterator(A, sub_batch), total=A//sub_batch, desc='AHAx ffts', leave=False):
                for l, u in tqdm(batch_iterator(C, coil_batch), total=C//coil_batch, desc='AHAx coils', leave=False):
                    x_coil = x[batch_slice + [slice(l, u), slice(v, w)]]
                    # Apply pad
                    x_coil = padder(x_coil)
                    # Apply Toeplitz'd NUFFT normal op
                    x_subsp_coil = 0
                    Fx = fft.fftn(x_coil.to(fft_device),
                                  dim=dim, norm='ortho')
                    Fx = Fx.unsqueeze(coildim + 1)
                    x_coil_sub = (self.oversamp_factor ** D) * fft.ifftn(
                        Fx * kernels[:, v:w].to(fft_device), dim=dim, norm='ortho'
                    ) # [C Aout Ain *2*im_size]
                    x_subsp_coil += torch.sum(x_coil_sub, dim=(-D-1)) # Sum over Ain

                    # Apply adjoint pad
                    x_coil = padder.adjoint(x_subsp_coil.to(x.device))

                    # Apply adjoint sensitivities
                    x_coil = x_coil * torch.conj(self.mps[l:u, None, ...])
                    out += torch.sum(x_coil, dim=(-D-2)) # Sum over coils
            return out

        def AHA_func(x: torch.Tensor):
            """
            x: [[N] A H W [D]]
            """
            out = torch.zeros_like(x)
            dim = tuple(range(-D, 0))
            # Apply sensitivies
            x = x.unsqueeze(coildim)
            x = x * self.mps[:, None, ...] # [[N] C Ain *im_size]
            for a_in in range(A):
                in_slc = batch_slice + [slice(None), a_in]
                x_a_in = x[in_slc] # [[N] C *im_size
                x_a_in = padder(x_a_in)
                for a_out in range(A):
                    kernel = kernels[a_out, a_in]
                    Fx_a_in = fft.fftn(
                        x_a_in,
                        dim=dim,
                        norm='ortho'
                    )
                    x_a_out = fft.ifftn(
                        Fx_a_in * kernel,
                        dim=dim,
                        norm='ortho'
                    )
                    x_a_out = fft.ifftshift(x_a_out, dim=dim)
                    x_a_out = padder.adjoint(x_a_out)
                    # apply adjoint coil
                    x_a_out *= torch.conj(self.mps)
                    out_slc = batch_slice + [a_out]
                    out[out_slc] += torch.sum(x_a_out, dim=coildim)
            return out


        return AHA_func, self.ishape, self.ishape

    def get_kernels(self, im_size, batch_size: Optional[int] = None):
        """Compute kernels with the current set of data
        batch_size: controls batching over trajectories
        """
        A, T = self.phi.shape
        device = self.phi.device
        dtype = torch.complex64
        batch_size = batch_size if batch_size is not None else T
        kernel_size = tuple(int(self.oversamp_factor*d) for d in im_size)
        kernels = torch.zeros((A, A, *kernel_size), dtype=dtype, device=device)
        for l, u in tqdm(
                batch_iterator(total=T, batch_size=batch_size),
                total=T//batch_size,
                desc='Computing toeplitz kernels',
                leave=False,
        ):
            trj_batch = self.trj[self.subsamp_idx, ...][:, l:u, ...] # [I Tsub D K]
            phi_batch = self.phi[:, l:u] # [A Tsub]
            sqrt_dcf_batch = self.sqrt_dcf[self.subsamp_idx, ...][:, l:u, ...] # [I Tsub K]

            kernels = toep._compute_weights_and_kernels(
                im_size,
                trj_batch,
                phi_batch,
                sqrt_dcf_batch,
                self.oversamp_factor,
                kernels,
                apply_scaling=False,
            )
        D = len(im_size)
        scale_factor = self.oversamp_factor/((np.prod(im_size)) ** (1/D))
        return kernels * scale_factor
