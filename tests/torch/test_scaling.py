import matplotlib.pyplot as plt
import sigpy as sp


def show_collection_of_images
    """Show a collection of subspace images with optional ROIs
    imgs_dict: Mapping from name -> [A H W] image
    roi_bbs: Optional mapping from name -> [A Hb Wb] ROI image
    """
    if len(imgs_dict) == 0:
        return

    imgs_list = list(imgs_dict.items())
    ncols = len(imgs_list[0][1])
    nrows = len(imgs_list)
    fig, axs = plt.subplots(ncols=ncols,
                            nrows=nrows,
                            figsize=(ncols, nrows))

    if nrows == 1:
        name, imgs = imgs_list[0]
        axs = plot_subspace_img(axs, imgs, name)
        if roi_bbs is not None and name in roi_bbs:
            for roi, roi_params in roi_bbs[name]:
                plot_bounding_box(axs, roi, roi_params)


    else:
        for axs, (name, imgs) in zip(axs, imgs_list):
            axs = plot_subspace_img(axs, imgs, name)
            if roi_bbs is not None and name in roi_bbs:
                for roi, roi_params in roi_bbs[name]:
                    plot_bounding_box(axs, roi, roi_params)
    return fig, axs

def test_scaling_AHA():
    device = get_device(device_idx)
    data = copy.deepcopy(rawdata)
    data = data.to_torch(device=device)
    data.mps = data.mps.to(torch.complex64) # Hack
    data = data.to_linop_shapes()
    data.print_shapes()
    linop_factory = SubspaceLinopFactory(
        trj=data.trj,
        phi=data.phi,
        mps=data.mps,
        sqrt_dcf=data.sqrt_dcf,
        oversamp_factor=1.30,
    )
    linop_factory.to(device)

    # Non-toeplitz version
    AHA, _, _ = linop_factory.get_normal()
    x2 = torch.from_numpy(x).to(device)
    AHAx2 = AHA(x2).detach().cpu().numpy()

    # Toeplitz version
    #@TorchCache.cache(Path(__file__).parent/'kernels'/'kernel.pth')
    def get_kernels():
        return linop_factory.get_kernels(
            data.get_shape('im_size'),
            batch_size=opt.mem_opt.toep['batch_size'],
        )
    kernels = get_kernels().to(device)
    AHA, _, _ = linop_factory.get_normal(
        kernels,
        batched=False,
        coil_batch=opt.mem_opt.normal['coil_batch'],
        sub_batch=opt.mem_opt.normal['sub_batch'],
        nufft_device=get_device(opt.mem_opt.normal['fft_device_idx']),
    )
    AHAx3 = AHA(x2).detach().cpu().numpy()

    imgs_dict = {
        'Original': AHAx1,
        'New (No Toep)': AHAx2,
        'New (Toep)': AHAx3,
    }
    for name, img in imgs_dict.items():
        print(f'{name}: min: {np.abs(img).min()}, max: {np.abs(img).max()}' )
    matplotlib.use('WebAgg')
    show_collection_of_subsp_imgs(imgs_dict)
    plt.show()


if __name__ == '__main__':
    test_scaling_AHA()
