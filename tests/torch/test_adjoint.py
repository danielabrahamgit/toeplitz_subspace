import numpy as np
import torch
import pytest

from toeplitz_subspace.torch.linop import SubspaceLinopFactory

def test_adjoint_subsamp(random_subsamp_2d_mrf_problem):
    trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img = random_subsamp_2d_mrf_problem
    linop_factory = SubspaceLinopFactory(
        trj, phi, mps, sqrt_dcf, subsamp_idx
    )
    AH, ishape, oshape = linop_factory.get_adjoint()
    assert ishape == ksp.shape
    x = AH(ksp)
    assert x.shape == oshape

def test_adjoint_fullsamp(random_fullsamp_2d_mrf_problem):
    trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img = random_fullsamp_2d_mrf_problem
    linop_factory = SubspaceLinopFactory(
        trj, phi, mps, sqrt_dcf, subsamp_idx
    )
    AH, ishape, oshape = linop_factory.get_adjoint()
    assert ishape == ksp.shape
    x = AH(ksp)
    assert x.shape == oshape
