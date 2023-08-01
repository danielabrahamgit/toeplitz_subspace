import numpy as np
import torch
import pytest

from toeplitz_subspace.torch.linop import SubspaceLinopFactory

def test_normal_subsamp(random_subsamp_2d_mrf_problem):
    trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img = random_subsamp_2d_mrf_problem
    linop_factory = SubspaceLinopFactory(
        trj, phi, mps, sqrt_dcf, subsamp_idx
    )
    AHA, ishape, oshape = linop_factory.get_normal()
    assert ishape == img.shape
    b = AHA(img)
    assert b.shape == oshape

    im_size = mps.shape[1:]
    kernels = linop_factory.get_kernels(im_size, batch_size=100)
    AHA_toep, ishape_toep, oshape_toep = linop_factory.get_normal_toeplitz(kernels, batched_input=False)
    assert ishape_toep == ishape
    b2 = AHA_toep(img)
    assert b2.shape == oshape_toep
    #assert torch.isclose(b, b2).all()

def test_normal_fullsamp(random_fullsamp_2d_mrf_problem):
    trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img = random_fullsamp_2d_mrf_problem
    linop_factory = SubspaceLinopFactory(
        trj, phi, mps, sqrt_dcf, subsamp_idx
    )
    AHA, ishape, oshape = linop_factory.get_normal()
    assert ishape == img.shape
    b = AHA(img)
    assert b.shape == oshape

    im_size = mps.shape[1:]
    kernels = linop_factory.get_kernels(im_size, batch_size=100)
    AHA_toep, ishape_toep, oshape_toep = linop_factory.get_normal_toeplitz(kernels, batched_input=False)
    assert ishape_toep == ishape
    b2 = AHA_toep(img)
    assert b2.shape == oshape_toep
    #assert torch.isclose(b, b2).all()

def inner(x: torch.Tensor, y: torch.Tensor):
    return torch.sum(torch.conj(x) * y)

def test_normal_hermitian(random_fullsamp_2d_mrf_problem):
    trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img = random_fullsamp_2d_mrf_problem
    linop_factory = SubspaceLinopFactory(
        trj, phi, mps, sqrt_dcf, subsamp_idx
    )
    AHA, ishape, oshape = linop_factory.get_normal()
    xAHAx = inner(img, AHA(img))

    assert (xAHAx.imag / xAHAx.real) < 1e-8

    A, _, _ = linop_factory.get_forward()
    AH, _, _ = linop_factory.get_adjoint()
    y = A(img)
    x2 = AH(y)
    xAHAx2 = inner(img, x2)

    assert torch.isclose(xAHAx2, xAHAx)
    assert torch.isclose(inner(y, y).real, xAHAx.real)

    im_size = mps.shape[1:]
    kernels = linop_factory.get_kernels(im_size, batch_size=100)
    AHA_toep, _, _= linop_factory.get_normal_toeplitz(kernels, batched_input=False)
    xAHAx_toep = inner(img, AHA_toep(img))

    assert (xAHAx_toep.imag / xAHAx_toep.real) < 1e-7
    assert torch.isclose(xAHAx_toep.real, xAHAx.real, rtol=1e-4)
