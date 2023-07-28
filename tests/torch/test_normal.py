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
