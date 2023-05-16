import numpy as np

def gen_FA_train(N_T):
    trs = np.arange(N_T)
    FA_train = np.maximum(90 - ((trs - 300)/10)**2, 30 - ((trs - 100)/25)**2)
    FA_train = np.maximum(10, FA_train)
    return FA_train

def gen_dict(TE, TR, FA_train, t1_range=None, t2_range=None, pd_range=None, params=None):

    if params is not None:
        t1s, t2s, pds = params
    else:
        # Gen parameters
        t1s = np.linspace(*t1_range)
        t2s = np.linspace(*t2_range)
        pds = np.linspace(*pd_range)
        t1s, t2s, pds = np.meshgrid(t1s, t2s, pds, indexing='ij')
        t1s = t1s.flatten()
        t2s = t2s.flatten()
        pds = pds.flatten()
        inds = np.argwhere(t2s < t1s)[:, 0]
        t1s = t1s[inds]
        t2s = t2s[inds]
        pds = pds[inds]

    # Loop over all TR
    N_T = len(FA_train)
    M = np.zeros((3, N_T, len(t1s)))
    for tr in range(1, N_T):
        print(f'TR = {tr+1}/{N_T}', end='\r', flush=True)
    
        # Start from last point
        pre_RF = M[:, tr-1, :]

        # RF excitation (rotate about x-axis)
        c = np.cos(FA_train[tr])    
        s = np.sin(FA_train[tr])    
        R_matrix = np.array([[1.,0.,0.],[0., c, s],[0,-s, c]])
        post_RF = np.sum(R_matrix[:, :, None] * pre_RF[None, ...], axis=1)

        # T2 Decay / T1 Recovery
        E1 = np.exp(-TE / t1s)
        E2 = np.exp(-TE / t2s)
        relax = post_RF * 0
        relax[:2, ...] = post_RF[:2, ...] * E2[None, ...]
        relax[2, ...] = post_RF[2, ...] * E1[None, ...] + (pds - E1[None, ...])

        # Record data
        M[:, tr, :] = relax

        # T2 Decay / T1 Recovery
        E1 = np.exp(-(TR - TE) / t1s)
        E2 = np.exp(-(TR - TE) / t2s)
        relax2 = relax * 0
        relax2[:2, ...] = relax[:2, ...] * E2[None, ...]
        relax2[2, ...] = relax[2, ...] * E1[None, ...] + (pds - E1[None, ...])
    
    return M[0, ...].T + 1j * M[1, ...].T
