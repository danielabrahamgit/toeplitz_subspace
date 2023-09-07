import numpy as np


__all__ = [
    'get_interleave_subsamp_idx',
    'subsamp_interleaves',
]

def ceil(num: int, denom: int):
    return -(num // -denom)

def get_interleave_subsamp_idx(total_interleaves,
                               n_timepoints,
                               n_interleaves_per_timepoint):
    """Helper function for generating subsampling interleave indices.

    In a trajectory with T TRs and I interleaves, to sample
    i interleaves per timepoint, want a sampling pattern like:
    TR idx:      [1 2 3 ...]
    Interlv idx: [1 2 3 ...]

    If multiple interleaves are necessary per TR, then need to space
    the multiple interleaves out. e.g. for 32 interleaves, need:
    TR idx:      [1      2      3 ...]
    Interlv idx: [[1 17] [2 18] [3 19] ...]
    """

    il_offset = total_interleaves // n_interleaves_per_timepoint
    idx = np.array(range(total_interleaves))
    idx = np.tile(idx, reps=ceil(n_timepoints, total_interleaves))
    idx = idx[:n_timepoints]

    # 1 interleave -> n interleaves per timepoint
    idx = np.stack([idx + i * il_offset
                    for i in range(n_interleaves_per_timepoint)])
    idx = idx % total_interleaves
    return idx


def subsamp_interleaves(trj, sqrt_dcf, ksp, n_interleaves=None, fix_ksp=True):
    """
    trj: [R 1 K D] or [R 1 D K] kspace trajectory
    - The 1 indicates that all R trajectories are used across all TRs
    sqrt_dcf: [R 1 K]
    ksp: [C R T K]
    """
    n_interleaves = n_interleaves if n_interleaves is not None else trj.shape[0]

    assert trj.shape[1] == 1, 'interleaves not shared across all TRs'
    assert sqrt_dcf.shape[1] == 1, 'interleaves not shared across all TRs'
    total_il = trj.shape[0]
    n_trs = ksp.shape[2]
    il_offset = total_il // n_interleaves

    idx = get_interleave_subsamp_idx(
        total_il, n_trs, n_interleaves,
    )

    # Subsample
    trj = trj[idx, 0, :, :]
    sqrt_dcf = sqrt_dcf[idx, 0, :]
    if fix_ksp:
        ksp = ksp[:, idx, np.arange(n_trs), :]

    return trj, sqrt_dcf, ksp, idx
