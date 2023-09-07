from einops import rearrange
import numpy as np
import torch
import pytest

from toeplitz_subspace.torch.linop import SubspaceLinopFactory
from toeplitz_subspace.torch.viz import plot_kernels_2d

def test_normal_subsamp(subsampled_spiralmrf2d):
    #trj, sqrt_dcf, mps, phi, subsamp_idx, ksp, img = random_subsamp_2d_mrf_problem
    trj, sqrt_dcf, ksp, phi, mps, img, subsamp_idx = subsampled_spiralmrf2d
    trj = torch.from_numpy(trj).float()
    sqrt_dcf = torch.from_numpy(sqrt_dcf).float()
    phi = torch.from_numpy(phi).type(torch.complex64)
    ksp = torch.from_numpy(ksp).type(torch.complex64)
    mps = torch.from_numpy(mps).type(torch.complex64)
    img = torch.from_numpy(img).type(torch.complex64)
    subsamp_idx = torch.from_numpy(subsamp_idx).long()

    # preprocess
    trj = trj[:, 0, ...] * 2*np.pi / img.shape[1]
    print(f'trj shape: {trj.shape}, {trj.dtype}')
    print(f'trj min: {trj.min()}, max: {trj.max()}')

    sqrt_dcf = sqrt_dcf[:, 0, :]
    print(f'sqrt_dcf shape: {sqrt_dcf.shape}, {sqrt_dcf.dtype}')

    ksp = rearrange(ksp, 'c i t k -> i t c k')
    print(f'ksp shape: {ksp.shape}, {ksp.dtype}')
    print(f'mps shape: {mps.shape}, {mps.dtype}')
    print(f'phi shape: {phi.shape}, {phi.dtype}')
    print(f'subsamp_idx shape: {subsamp_idx.shape}, {subsamp_idx.dtype}')
    linop_factory = SubspaceLinopFactory(
        trj, phi, mps, sqrt_dcf, subsamp_idx,
        oversamp_factor=4.
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

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('WebAgg')
    fig, ax = plt.subplots(nrows=2, ncols=img.shape[0])
    if img.shape[0] == 1:
        pcm = ax[0].imshow(
            np.rot90(np.abs(b.detach().cpu().numpy()), axes=(-2, -1))[0]
        )
        fig.colorbar(pcm, ax=ax[0], orientation='vertical', location='right')
        pcm = ax[1].imshow(
            np.rot90(np.abs(b2.detach().cpu().numpy()), axes=(-2, -1))[0]
        )
        fig.colorbar(pcm, ax=ax[1], orientation='vertical', location='right')

    plot_kernels_2d(kernels)

    plt.show()




    assert b2.shape == oshape_toep
    assert torch.isclose(b, b2).all()

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
