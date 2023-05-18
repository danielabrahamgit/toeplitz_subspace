import torch
import torch.nn as nn
import torch.nn.functional as F

class PadLast(nn.Module):
    def __init__(self, pad_im_size, im_size):
        super().__init__()
        assert len(pad_im_size) == len(im_size)
        self.im_dim = len(im_size)
        self.im_size = tuple(im_size)
        self.pad_im_size = tuple(pad_im_size)

        sizes = [[(psz - isz) // 2]*2 for psz, isz in zip(pad_im_size, im_size)]
        self.pad = sum(sizes, start=[])
        self.pad.reverse()

        self.crop_slice = [slice(self.pad[2*i], -self.pad[2*i+1])
                           for i in range(len(self.pad)//2)]
        self.crop_slice.reverse()

    def forward(self, x):
        """Pad the last n dimensions of x"""
        assert tuple(x.shape[-self.im_dim:]) == self.im_size
        pad = self.pad + [0, 0]*(x.ndim - self.im_dim)
        return F.pad(x, pad)

    def adjoint(self, y):
        """Crop the last n dimensions of y"""
        assert tuple(y.shape[-self.im_dim:]) == self.pad_im_size
        slc = [slice(None)] * (y.ndim - self.im_dim) + self.crop_slice
        return y[slc]

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

if __name__ == '__main__':
    test_pad_last()
