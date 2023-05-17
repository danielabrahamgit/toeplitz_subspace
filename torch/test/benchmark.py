import sys
sys.path.append('..')
from collections import OrderedDict

from einops import rearrange
import matplotlib
import matplotlib.pyplot as plt
import torch

from linop import SubspaceLinopFactory
from invivo_data import load_data




def test_mrf_subsp_linop():
    # Load data
    device = torch.device(
        f'cuda' if torch.cuda.is_available() else 'cpu'
    )
    ksp, trj, dcf, phi, mps = load_data(device, verbose=True)
    linop_factory = SubspaceLinopFactory(
        trj, phi, mps, torch.sqrt(dcf)
    )

    A, ishape, oshape = linop_factory.get_forward()
    AH, _, _ = linop_factory.get_adjoint()
    AHA, _, _ = linop_factory.get_normal(
        toeplitz=True,
        device=device,
        verbose=True,
    )

    # Try stuff
    y = rearrange(ksp, 't c k ->  wl  ')
    AHb = AH(y)
    yhat = A(AHb)

    # Compare Naive and Toeplitz
    im_size = linop_factory.ishape[-2:]
    A = linop_factory.ishape[0]
    x = torch.tensor(sp.shepp_logan(im_size)).type(torch.complex64)
    x = x.repeat(A, 1, 1)
    # zero out later coeffs
    x[1:, :, :] = 0.
    x = x.to(device)

    AHA_naive, _, _ = linop_factory.get_normal(toeplitz=False)
    AHA_embed, _, _ = linop_factory.get_normal(toeplitz=True, device=device)


    outputs = OrderedDict()
    timings = OrderedDict()
    outputs['naive'] = AHA_naive(x).detach().cpu().numpy()
    outputs['toeplitz'] = AHA_embed(x).detach().cpu().numpy()

    reporter = ptm.MemReporter()
    reporter.report

    # Plotting
    matplotlib.use('WebAgg')
    nrows = len(outputs)
    fig, ax = plt.subplots(nrows=nrows, ncols=A)
    for r, (title, img_out) in enumerate(outputs.items()):
        for c in range(A):
            pcm = ax[r, c].imshow(np.abs(img_out[c]))
            fig.colorbar(pcm, ax=ax[r, c], location='right')
            if c == 0:
                ax[r, c].set_ylabel(title)
            if r == 0:
                ax[r, c].set_title(f'Coeff {c}')
            turn_off_ticks_but_keep_labels(ax[r, c])
    fig.suptitle('Image recons')

    plt.show()

if __name__ == '__main__':
    test_mrf_subsp_linop()
