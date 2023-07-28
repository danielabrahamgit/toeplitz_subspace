from pathlib import Path
import sys
from collections import OrderedDict
import logging

from einops import rearrange
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_memlab as ptm
import torch
import sigpy as sp
import tyro

sys.path.append(str(Path(__file__).parent.parent))
from linop import SubspaceLinopFactory
from invivo_data import load_data
from timing import tictoc
from utils import setup_console_logger
from caching import TorchCache

logger = logging.getLogger()

def get_device(device_idx: int):
    return torch.device(
        f'cuda:{device_idx}' if device_idx >= 0 else 'cpu'
    )

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


def test_mrf_subsp_linop(device_idx: int = -1):
    device = get_device(device_idx)
    setup_console_logger(logger, logging.INFO)
    logger.info(f'Using {device}')

    ksp, trj, dcf, phi, mps = load_data(device, verbose=True)
    T = ksp.shape[0]
    im_size = mps.shape[1:]
    #R = 30
    #trj_subsamp = trj[:R] # Later trajectories repeat
    #dcf_subsamp = dcf[:R]
    #subsamp_idx = torch.arange(R).repeat(T//R)
    linop_factory = SubspaceLinopFactory(
        trj, phi, mps, torch.sqrt(dcf),
    )
    linop_factory.to(device)

    A, ishape, oshape = linop_factory.get_forward(coil_batch=2)
    AH, _, _ = linop_factory.get_adjoint(coil_batch=2)

    # Try stuff
    y = ksp * torch.sqrt(dcf)[:, None, :]
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
        trj,
        phi,
        mps,
        torch.sqrt(dcf),
    )
    linop_factory.to(device)

    AHA_naive, _, _ = linop_factory.get_normal(coil_batch=2)
    logger.info('Computing kernels...')
    @TorchCache.cache(Path(__file__).parent/'kernel.npz')
    def get_kernels():
        return linop_factory.get_kernels(im_size)
    kernels = get_kernels()
    AHA_embed, _, _ = linop_factory.get_normal(kernels, batched=False)

    outputs = OrderedDict()
    timings = OrderedDict()
    logger.info('Computing naive normal')
    outputs['naive'] = AHA_naive(x).detach().cpu().numpy()
    logger.info('Computing toeplitz normal')
    outputs['toeplitz'] = AHA_embed(x).detach().cpu().numpy()

    n_trials = 1
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
    tyro.cli(test_mrf_subsp_linop)
