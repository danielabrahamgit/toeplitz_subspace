import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sigpy as sp
import torch
import torch.fft as fft

def test_fft_ifft():
    img = sp.shepp_logan((64, 64))
    img = torch.from_numpy(img).type(torch.complex64)
    Fimg = fft.fftn(img, dim=(-2, -1), norm='ortho')
    img2 = fft.ifftn(Fimg, dim=(-2, -1), norm='ortho')
    matplotlib.use('WebAgg')
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].imshow(np.abs(img.detach().cpu().numpy()))
    ax[0].set_title('Original image')
    ax[1].imshow(np.abs(Fimg.detach().cpu().numpy()))
    ax[1].set_title('FFT(img)')
    ax[2].imshow(np.abs(img2.detach().cpu().numpy()))
    ax[2].set_title('iFFT(FFT(img))')
    #plt.show()
    assert torch.isclose(img, img2, atol=1e-6).all()

def test_fft_filter_ifft():
    filt = torch.zeros((64, 64)).type(torch.complex64)
    filt[32, 32] = 1.
    filt = fft.ifftshift(filt, (0, 1))
    img = sp.shepp_logan((64, 64))
    img = torch.from_numpy(img).type(torch.complex64)

    Fimg = fft.fftn(img, dim=(-2, -1), norm='ortho')
    Ffilt = fft.fftn(filt, dim=(-2, -1))
    img2 = fft.ifftn(Fimg*Ffilt, dim=(-2, -1), norm='ortho')
    matplotlib.use('WebAgg')
    fig, ax = plt.subplots(nrows=1, ncols=6)
    ax[0].imshow(np.abs(img.detach().cpu().numpy()))
    ax[0].set_title('Original image')
    ax[1].imshow(np.abs(filt.detach().cpu().numpy()))
    ax[1].set_title('Filter')
    ax[2].imshow(np.abs(Fimg.detach().cpu().numpy()))
    ax[2].set_title('FFT(img)')
    ax[3].imshow(np.abs((Fimg * Ffilt).detach().cpu().numpy()))
    ax[3].set_title('FFT(img) * FFT(filt)')
    ax[4].imshow(np.abs(img2.detach().cpu().numpy()))
    ax[4].set_title('iFFT(FFT(img) * FFT(filt))')
    ax[5].imshow(np.abs((img - img2).detach().cpu().numpy()))
    ax[5].set_title('diff')
    #plt.show()
    assert torch.isclose(img, img2, atol=1e-6).all()
