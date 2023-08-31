import sys
sys.path.append('..')
from collections import OrderedDict

from einops import rearrange
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_memlab as ptm
import torch
import sigpy as sp

from linop import SubspaceLinopFactory
from invivo_data import load_data
from timing import tictoc


def turn_off_ticks_but_keep_labels(ax):
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    return ax

def torch_benchmark(fn, n_trials, *args, **kwargs):
    timings = []
    for i in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn(*args, **kwargs)
        end.record()

        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        print(f'Trial {i}: {time}')
        timings.append(time)
    return timings


def test_mrf_subsp_linop():
    # Load data
    # device = torch.device(
    #     f'cuda' if torch.cuda.is_available() else 'cpu'
    # )
    device = torch.device('cpu')
    print(f'Using {device}')
    ksp, trj, dcf, phi, mps = load_data(device, verbose=True)
    T = ksp.shape[0]
    #R = 30
    #trj_subsamp = trj[:R] # Later trajectories repeat
    #dcf_subsamp = dcf[:R]
    #subsamp_idx = torch.arange(R).repeat(T//R)
    linop_factory = SubspaceLinopFactory(
        trj, phi, mps, torch.sqrt(dcf),
    )
    linop_factory.to(device)

    A, ishape, oshape = linop_factory.get_forward()
    AH, _, _ = linop_factory.get_adjoint()

    # Try stuff
    y = ksp
    with tictoc(name='adjoint', verbose=True):
        AHb = AH(y)
    with tictoc(name='forward', verbose=True):
        yhat = A(AHb)

    # Compare Naive and Toeplitz
    linop_factory.to(device)
    im_size = linop_factory.ishape[-2:]
    A = linop_factory.ishape[0]
    x = torch.tensor(sp.shepp_logan(im_size)).type(torch.complex64)
    x = x.repeat(A, 1, 1)
    # zero out later coeffs
    x[1:, :, :] = 0.
    x = x.to(device)
    linop_factory = SubspaceLinopFactory(
        trj_subsamp, phi, torch.ones_like(mps).type(torch.complex64),
        torch.sqrt(dcf_subsamp), subsamp_idx,
    )
    linop_factory.to(device)

    AHA_naive, _, _ = linop_factory.get_normal(toeplitz=False)
    AHA_embed, _, _ = linop_factory.get_normal(toeplitz=True,
                                               device=device,
                                               verbose=True)

    outputs = OrderedDict()
    timings = OrderedDict()
    outputs['naive'] = AHA_naive(x).detach().cpu().numpy()
    outputs['toeplitz'] = AHA_embed(x).detach().cpu().numpy()

    n_trials = 5
    timings['naive'] = torch_benchmark(AHA_naive, n_trials, x)
    timings['toeplitz'] = torch_benchmark(AHA_embed, n_trials, x)


    reporter = ptm.MemReporter()
    reporter.report()

    # Plotting
    matplotlib.use('WebAgg')
    x = x.detach().cpu().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=A)
    for a in range(A):
        ax[a].imshow(np.abs(x[a, ...]))
    plt.title('Input')

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

    # Timings
    timings_arr = np.array([v for k, v in timings.items()])
    medians = np.median(timings_arr, axis=1)
    q25_q75 = np.quantile(timings_arr, q=[0.25, 0.75], axis=1)
    q25_q75_rel = q25_q75 - medians
    q25_q75_rel[0, :] = -q25_q75_rel[0, :] # Flip sign of lower error
    plt.figure()
    plt.bar(np.arange(len(timings)), medians, yerr=q25_q75_rel,
            tick_label=list(timings.keys()))
    plt.yscale('log')
    plt.ylabel('Time (ms)')
    plt.xlabel('Method')
    plt.title('Median and Q25-Q75 Timings')

    plt.show()

if __name__ == '__main__':
    test_mrf_subsp_linop()
