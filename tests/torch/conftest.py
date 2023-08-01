from einops import rearrange
import numpy as np
import sigpy as sp
import sigpy.mri as mri
import torch
import pytest


@pytest.fixture
def vds():
    trj = mri.spiral(fov=1, N=110, f_sampling=1, R=1, ninterleaves=16, alpha=1.1, gm=0.4, sm=10)
    trj = rearrange(trj, '(r t) d -> r d t', r=16)
    trj = torch.from_numpy(trj) * (2*np.pi/110)
    trj = trj.float()
    return trj
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('WebAgg')
    # plt.plot(trj[1, ..., 0], trj[1, ..., 1])
    # plt.show()
    # breakpoint()

@pytest.fixture(params=[1, 8])
def random_subsamp_2d_mrf_problem(request, vds):
    I = request.param
    R = 16
    T = 100
    C = 12
    A = 5
    im_size = (110, 110)
    D = len(im_size)

    #trj = torch.zeros((R, D, K)).uniform_(-np.pi, np.pi)
    trj = vds
    K = vds.shape[-1]
    sqrt_dcf = torch.randn((R, K))
    mps = torch.randn((C, *im_size)) + 1j * torch.randn((C, *im_size))
    phi = torch.randn((A, T))
    subsamp_idx = torch.randint(R, size=(I, T))

    ksp = torch.randn((I, T, C, K)) + 1j*torch.randn((I, T, C, K))
    #img = torch.randn((A, *im_size)) + 1j*torch.randn((A, *im_size))
    img = torch.zeros((A, *im_size)).type(torch.complex64)
    img[0] = torch.from_numpy(sp.shepp_logan(im_size)).type(torch.complex64)
    return (trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img)


@pytest.fixture
def random_fullsamp_2d_mrf_problem(vds):
    R, I = 16, 16
    T = 100
    C = 12
    A = 5
    im_size = (110, 110)
    D = len(im_size)

    K = vds.shape[-1]
    trj = rearrange(vds, 'r d k -> 1 d (r k)')

    #trj = torch.zeros((1, D, R*K)).uniform_(-np.pi, np.pi)
    #sqrt_dcf = torch.randn((1, R*K))
    sqrt_dcf = torch.ones((1, R*K))
    #mps = torch.randn((C, *im_size)) + 1j * torch.randn((C, *im_size))
    #mps = torch.randn((C, *im_size))
    mps = torch.ones((C, *im_size))
    mps = mps.type(torch.complex64)
    #phi = torch.randn((A, T))
    phi = torch.ones((A, T))
    subsamp_idx = None

    ksp = torch.randn((1, T, C, I*K)) + 1j*torch.randn((1, T, C, R*K))
    #img = torch.randn((A, *im_size)) + 1j*torch.randn((A, *im_size))
    #img = torch.randn((A, *im_size))
    img = torch.zeros((A, *im_size)).type(torch.complex64)
    img[0] = torch.from_numpy(sp.shepp_logan(im_size)).type(torch.complex64)
    img = img.type(torch.complex64)
    return (trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img)
