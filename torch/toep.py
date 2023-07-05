from typing import Tuple

import torch
from torchkbnufft import calc_toeplitz_kernel


def compute_weights(
        subsamp_idx: torch.Tensor,
        phi: torch.Tensor,
        sqrt_dcf: torch.Tensor,
        memory_efficient: bool = False,
):
    device = phi.device
    dtype = phi.dtype
    R, K = sqrt_dcf.shape
    A, T = phi.shape
    weights = torch.zeros((R, A, A, K), device=device).type(dtype)
    subsamp_mask = torch.zeros((R, T), device=device).type(dtype)
    subsamp_mask[subsamp_idx, torch.arange(T)] = 1.
    for a_in in range(A):
        weight = torch.zeros((R, A, K), device=device).type(dtype)
        weight[:, a_in, :] = 1.
        if memory_efficient:
            out = []
            for r in range(R):
                idx = subsamp_idx[r]
                tmp = torch.einsum('at,ak->tk', phi, weight[idx])
                tmp = tmp * (sqrt_dcf[r] ** 2)
                tmp = torch.einsum('at,tk->ak', torch.conj(phi), weight[idx])
                out.append(tmp)
            weight = torch.stack(out, dim=0)
        else:
            weight = torch.einsum('at,rak->rtk', phi, weight)
            weight = weight * subsamp_mask[..., None] * (sqrt_dcf[:, None, :] ** 2)
            weight = torch.einsum('at,rtk->rak', torch.conj(phi), weight)
        weights[:, :, a_in, :] = weight
    return weights


def compute_kernels(
        trj: torch.Tensor,
        weights: torch.Tensor,
        im_size: Tuple,
        oversamp_factor: float,
        device='cpu',
        verbose=False,
):
    A = weights.shape[1]
    kernel_size = (oversamp_factor * d for d in im_size)
    kernels = torch.zeros((A, A, *kernel_size), device=device).type(weights.dtype)
    for a_in in range(A):
        for a_out in range(A):
            if verbose:
                print(f'>> Calculating kernel({a_out}, {a_in})')
            kernel = calc_toeplitz_kernel(
                trj,
                im_size,
                weights=weights[:, None, a_out, a_in, :],
            )
            kernels[a_out, a_in, ...] = kernel.sum(dim=0)
    return kernels
