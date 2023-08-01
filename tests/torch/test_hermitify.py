import pytest
import sigpy as sp
import torch
import torch.fft as fft

from toeplitz_subspace.torch.toep import hermitify

def test_hermitify():
    img = sp.shepp_logan((5, 4))
    img = torch.from_numpy(img).type(torch.complex64)
    Fimg = fft.fftn(img, dim=(0, 1))

    Fimg = hermitify(Fimg, ndims=2)

    img2 = fft.ifftn(Fimg, dim=(0, 1))
    assert torch.isclose(img, img2, atol=1e-7).all()
