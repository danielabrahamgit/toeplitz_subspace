import numpy as np
import torch
import pytest

from toeplitz_subspace.torch.linop import SubspaceLinopFactory

def test_forward_subsamp(random_subsamp_2d_mrf_problem):
    trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img = random_subsamp_2d_mrf_problem
    linop_factory = SubspaceLinopFactory(
        trj, phi, mps, sqrt_dcf, subsamp_idx
    )
    A, ishape, oshape = linop_factory.get_forward()
    assert ishape == img.shape
    b = A(img)
    assert b.shape == oshape

def test_forward_fullsamp(random_fullsamp_2d_mrf_problem):
    trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img = random_fullsamp_2d_mrf_problem
    linop_factory = SubspaceLinopFactory(
        trj, phi, mps, sqrt_dcf, subsamp_idx
    )
    A, ishape, oshape = linop_factory.get_forward()
    assert ishape == img.shape
    b = A(img)
    assert b.shape == oshape
