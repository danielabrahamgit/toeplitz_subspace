import torch
from toeplitz_subspace.torch.pad import PadLast


def test_pad_last():
    pad_im_size = (20, 40)
    im_size = (10, 20)

    padder = PadLast(pad_im_size, im_size)
    print(padder.pad)
    print(padder.crop_slice)

    x = torch.ones((2, 1, *im_size))
    y = padder(x)
    assert y.shape == (2, 1, *pad_im_size)

    xhat = padder.adjoint(y)
    assert x.shape == (2, 1, *im_size)
