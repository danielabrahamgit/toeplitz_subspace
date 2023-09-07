from einops import rearrange
import numpy as np
import sigpy as sp
import sigpy.mri as mri
import torch
import pytest
from tqdm import tqdm

from subsample import subsamp_interleaves


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

def spiral(im_size, R, n_shots, alpha=1.5):
    trj = mri.spiral(
        fov=1,
        N=im_size[0],
        f_sampling=0.8, # TODO function of self.n_read
        R=R,
        ninterleaves=n_shots,
        alpha=alpha,
        gm=40e-3, # Tesla / m
        sm=100, # Tesla / m / s
    )
    assert trj.shape[0] % n_shots == 0
    trj = trj.reshape((trj.shape[0] // n_shots, n_shots, 2), order='F')
    return trj

@pytest.fixture
def spiralmrf2d():
    n = 256 # image dim
    im_size = (n, n) # image dim
    nc = 16 # number of coils
    sigma = 1e-2 * 0 # Noise std
    trj_type = 'spiral' # trajectory type
    subspace_dim = 1
    n_trs = 1
    n_shots = 4
    #max_iter = 20 # max CG iterations
    #R = 3 # Undersampling rate

    # Subspace basis
    M = np.random.randn(n_trs, subspace_dim) + 1j * np.random.randn(n_trs, subspace_dim)
    Q, R = np.linalg.qr(M, mode='reduced')
    phi = Q.T.real

    # Phantom and sensitivity maps
    phantom = sp.shepp_logan(im_size)
    phantom = np.tile(phantom, reps=(subspace_dim, 1, 1))
    phantom *= np.random.randn(subspace_dim, 1, 1)
    mps = mri.birdcage_maps((nc, *im_size))

    trj = spiral(im_size, R=1, n_shots=n_shots, alpha=1.5)
    dcf = mri.pipe_menon_dcf(trj, im_size)
    dcf /= np.max(dcf)
    sqrt_dcf = np.sqrt(dcf)
    #sqrt_dcf = np.tile(sqrt_dcf, reps=(n_shots, 1, 1))

    # Forward models for ksp simulation
    img = np.einsum('at,a...->t...', phi, phantom)
    A = mri.linop.Sense(mps, trj)
    ksp = []
    for time_img in tqdm(img, total=img.shape[0]):
        ksp.append(A * time_img)
    ksp = np.stack(ksp, axis=0) # T C K
    ksp += sigma * np.random.randn(*ksp.shape) + 1j * sigma * np.random.randn(*ksp.shape)

    # Fix shapes
    trj = rearrange(trj, 'k r d -> r 1 d k')
    ksp = rearrange(ksp, 't c k r -> c r t k')
    sqrt_dcf = rearrange(sqrt_dcf, 'k r -> r 1 k')

    # Show data shapes
    print(f'trj shape = {trj.shape}')
    print(f'sqrt_dcf shape = {sqrt_dcf.shape}')
    print(f'ksp shape = {ksp.shape}')
    print(f'mps shape = {mps.shape}')
    print(f'phi shape = {phi.shape}')
    return trj, sqrt_dcf, ksp, phi, mps, phantom

@pytest.fixture
def subsampled_spiralmrf2d(spiralmrf2d):
    n_interleaves = 1
    trj, sqrt_dcf, ksp, phi, mps, phantom = spiralmrf2d
    _, _, ksp, idx = subsamp_interleaves(trj, sqrt_dcf, ksp, n_interleaves=n_interleaves)
    print(f'trj shape = {trj.shape}')
    print(f'sqrt_dcf shape = {sqrt_dcf.shape}')
    print(f'ksp shape = {ksp.shape}')
    print(f'mps shape = {mps.shape}')
    print(f'phi shape = {phi.shape}')
    print(f'idx shape = {idx.shape}')
    return trj, sqrt_dcf, ksp, phi, mps, phantom, idx


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
