import torch




def test_mrf_subsp_linop():
    # Load data
    device = torch.device(
        f'cuda' if torch.cuda.is_available() else 'cpu'
    )
    fullsamp_sp = load_fullysampled_data()
    fullsamp = fullsamp_sp.to_torch(device=device)
    subsamp_config = SubsampleConfig(
        interleaves_per_timepoint=1
    )
    subsamp = subsample(fullsamp_sp, **asdict(subsamp_config))
    subsamp = subsamp.to_torch(device=device)

    linop_factory = MRFSubspaceLinopFactory(fullsamp, subsamp_config)
    linop_factory.to(device)
    A_func, ishape, oshape = linop_factory.get_forward()
    AH_func, _, _ = linop_factory.get_adjoint()

    # Try stuff
    y = rearrange(subsamp.y, 'c i t k -> i t c k')
    AHb = AH_func(y)
    yhat = A_func(AHb)

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
