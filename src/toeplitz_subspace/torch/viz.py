import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

def turn_off_ticks_but_keep_labels(ax):
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    return ax

def plot_kernels_2d(kernels):
    fig = plt.figure()
    nrows, ncols = kernels.shape[:2]
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            ax = grid[ncols*i + j]
            ax.imshow(np.fft.fftshift(np.abs(kernels[i, j].detach().cpu().numpy()), axes=(-2, -1)))
            if i == 0:
                ax.set_title(f'in[{j}]')
            if j == 0:
                ax.set_ylabel(f'out[{i}]')
            turn_off_ticks_but_keep_labels(ax)
    fig.suptitle('Kernels')
    return fig
