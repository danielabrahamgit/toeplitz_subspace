from typing import Tuple, Optional, Union
import logging

import numpy as np
import torch
import torch.fft as fft
from torchkbnufft import calc_toeplitz_kernel, KbNufftAdjoint

logger = logging.getLogger(__name__)


def compute_weights(
        subsamp_idx: torch.Tensor,
        phi: torch.Tensor,
        sqrt_dcf: torch.Tensor,
        memory_efficient: bool = False,
):
    if memory_efficient:
        return _compute_weights_mem_eff(subsamp_idx, phi, sqrt_dcf)
    return _compute_weights(subsamp_idx, phi, sqrt_dcf)
    return _compute_weights_alt(subsamp_idx, phi, sqrt_dcf)

def _compute_weights_mem_eff(
        subsamp_idx: torch.Tensor,
        phi: torch.Tensor,
        sqrt_dcf: torch.Tensor,
):
    device = phi.device
    dtype = phi.dtype
    R, K = sqrt_dcf.shape
    A, T = phi.shape
    weights = []
    for a_in in range(A):
        input_ = torch.zeros((A, K), device=device, dtype=dtype)
        input_[a_in] = 1.
        out = []
        for r in range(R):
            tmp = torch.einsum('at,ak->tk', phi, input_)
            tmp = tmp * (sqrt_dcf[r] ** 2)
            tmp = torch.einsum('at,tk->ak', torch.conj(phi), tmp)
            out.append(tmp)
        weight = torch.stack(out, dim=0)
        weights.append(weight)
    return torch.stack(weights, dim=1)

def _compute_weights(
        subsamp_idx: torch.Tensor,
        phi: torch.Tensor,
        sqrt_dcf: torch.Tensor,
):
    device = phi.device
    dtype = phi.dtype
    R, K = sqrt_dcf.shape
    A, T = phi.shape
    weights = torch.zeros((R, A, A, K), dtype=dtype, device=device)
    subsamp_mask = torch.zeros((R, T), device=device).type(dtype)
    subsamp_mask[subsamp_idx, torch.arange(T)] = 1.
    for a_in in range(A):
        weight = torch.zeros((R, A, K), device=device).type(dtype)
        weight[:, a_in, :] = 1.
        weight = torch.einsum('at,rak->rtk', phi, weight)
        weight = weight * subsamp_mask[..., None] * (sqrt_dcf[:, None, :] ** 2)
        weight = torch.einsum('at,rtk->rak', torch.conj(phi), weight)
    weights[:, :, a_in, :] = weight
    return weights

def _compute_weights_and_kernels(
        im_size: Tuple,
        trj: torch.Tensor,
        subsamp_idx: torch.Tensor,
        phi: torch.Tensor,
        sqrt_dcf: torch.Tensor,
        oversamp_factor: float = 2.,
):
    """Doing this myself since calc_toeplitz_kernel was being strange
    """
    device = phi.device
    dtype = phi.dtype
    D = len(im_size)
    R, K = sqrt_dcf.shape
    A, T = phi.shape
    kernel_size = tuple(int(oversamp_factor*d) for d in im_size)
    kernels = torch.zeros((A, A, *kernel_size), dtype=dtype, device=device)
    adj_nufft = KbNufftAdjoint(kernel_size, grid_size=kernel_size, device=device)
    for a_in in range(A):
        for a_out in range(A):
            weight = torch.ones((R, K), dtype=dtype, device=device).type(dtype)
            weight *= sqrt_dcf
            weight = weight[subsamp_idx, ...] # [T K]
            weight *= phi[a_in][:, None] * torch.conj(phi[a_out][:, None])
            # Note: this sqrt dcf should go before the summation
            weight *= sqrt_dcf[subsamp_idx, :]
            weight = torch.zeros((R, K), dtype=dtype, device=device).index_add_(0, subsamp_idx, weight)
            # Adj nufft with all-ones sensitivity maps
            kernel = adj_nufft(
                weight[:, None, :],
                trj,
                smaps=torch.ones((1, *kernel_size), dtype=dtype, device=device)
            )
            # Summing out R and (fake) C dimension
            kernels[a_out, a_in] = fft.fftn(kernel.sum((0, 1)), dim=tuple(range(-D, 0)), norm='ortho')
    # Feels suspicious... need to test in 3D
    scale_factor = oversamp_factor/((np.prod(im_size)) ** (1/D))
    return kernels * scale_factor

def compute_kernels(
        trj: torch.Tensor,
        weights: torch.Tensor,
        im_size: Tuple,
        oversamp_factor: float = 2,
        kernels_device: Optional[Union[str, torch.device]] = 'cpu',
):
    """
    kernel_device: use cpu if memory is an issues
    """
    device = weights.device
    A = weights.shape[1]
    kernel_size = tuple(int(oversamp_factor*d) for d in im_size)
    kernel_half_size = tuple(int(oversamp_factor*d/2) for d in im_size)
    kernels = torch.zeros((A, A, *kernel_size),
                          dtype=weights.dtype,
                          device=kernels_device)
    for a_in in range(A):
        for a_out in range(A):
            logger.info(f'Calculating kernel({a_out}, {a_in})')
            kernel = calc_toeplitz_kernel(
                trj,
                im_size=kernel_half_size,
                weights=weights[:, None, a_out, a_in, :],
                grid_size=kernel_size,
            )
            kernels[a_out, a_in, ...] = kernel.sum(dim=0).to(kernels_device)
    return kernels

def compute_kernels_alt(
        trj: torch.Tensor,
        weights: torch.Tensor,
        im_size: Tuple,
        oversamp_factor: float = 2.,
        kernels_device: Optional[Union[str, torch.device]] = 'cpu',
):
    device = weights.device
    A = weights.shape[1]
    for a_in in range(A):
        for a_out in range(A):
            logger.info(f'Calculating kernel({a_out}, {a_in})')
            kernel = calc_toeplitz_kernel(
                trj,
                im_size=kernel_half_size,
                weights=weights[:, None, a_out, a_in, :],
                grid_size=kernel_size,
            )
            kernels[a_out, a_in, ...] = kernel.sum(dim=0).to(kernels_device)
