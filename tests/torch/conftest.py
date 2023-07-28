import numpy as np
import torch
import pytest

@pytest.fixture(params=[1, 8])
def random_subsamp_2d_mrf_problem(request):
    I = request.param
    R = 16
    T, K = 100, 800
    C = 12
    A = 5
    im_size = (110, 110)
    D = len(im_size)

    trj = torch.zeros((R, D, K)).uniform_(-np.pi, np.pi)
    sqrt_dcf = torch.randn((R, K))
    mps = torch.randn((C, *im_size)) + 1j * torch.randn((C, *im_size))
    phi = torch.randn((A, T))
    subsamp_idx = torch.randint(R, size=(I, T))

    ksp = torch.randn((I, T, C, K)) + 1j*torch.randn((I, T, C, K))
    img = torch.randn((A, *im_size)) + 1j*torch.randn((A, *im_size))
    return (trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img)


@pytest.fixture
def random_fullsamp_2d_mrf_problem():
    R, I = 16, 16
    T, K = 100, 800
    C = 12
    A = 5
    im_size = (110, 110)
    D = len(im_size)

    trj = torch.zeros((1, D, R*K)).uniform_(-np.pi, np.pi)
    sqrt_dcf = torch.randn((1, R*K))
    mps = torch.randn((C, *im_size)) + 1j * torch.randn((C, *im_size))
    phi = torch.randn((A, T))
    subsamp_idx = None

    ksp = torch.randn((1, T, C, I*K)) + 1j*torch.randn((1, T, C, R*K))
    img = torch.randn((A, *im_size)) + 1j*torch.randn((A, *im_size))
    return (trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img)
