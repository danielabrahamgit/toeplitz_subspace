from typing import Tuple, Optional, Union
import logging

import torch
from torchkbnufft import calc_toeplitz_kernel

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
