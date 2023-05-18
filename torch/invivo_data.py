from pathlib import Path
import subprocess

from einops import rearrange
import numpy as np
import torch
import gdown

def download():
    folder_url = 'https://drive.google.com/drive/folders/12IiApk8485EnmA0oe9doif9dg1oOxFOL'
    datapath = Path('./data')
    if not datapath.is_dir():
        # Download and extract
        gdown.download_folder(folder_url, quiet=False, use_cookies=False)
        subprocess.Popen('unzip mrf_2d_data/data.zip', shell=True).wait()
        subprocess.Popen('rm -rf mrf_2d_data', shell=True).wait()
    else:
        print(f'{datapath} already exists')


def load_data(device='cpu', verbose=False):
    data = Path(__file__).parent/'data'
    ksp = torch.from_numpy(np.load(data/'ksp.npy')).to(device) # Raw k-space data
    trj = torch.from_numpy(np.load(data/'trj.npy')).type(torch.float32).to(device) # Trajectory
    dcf = torch.from_numpy(np.load(data/'dcf.npy')).type(torch.float32).to(device) # Density compensation
    phi = torch.from_numpy(np.load(data/'phi.npy')).type(torch.complex64).to(device) # Subspace basis
    mps = torch.from_numpy(np.load(data/'mps.npy')).type(torch.complex64).to(device) # Sensitivity maps

    # Fix dimensions
    ksp = rearrange(ksp, 'c k 1 t -> t c k')
    trj = rearrange(trj, 'k 1 r d -> r d k') * 2 * np.pi
    dcf = rearrange(dcf, 'k 1 r -> r k')

    if verbose:
        # Show dimensions and types
        print(f'ksp shape = {ksp.shape}, dtype = {ksp.dtype}')
        print(f'trj shape = {trj.shape}, dtype = {trj.dtype}')
        print(f'dcf shape = {dcf.shape}, dtype = {dcf.dtype}')
        print(f'phi shape = {phi.shape}, dtype = {phi.dtype}')
        print(f'mps shape = {mps.shape}, dtype = {mps.dtype}')
    return ksp, trj, dcf, phi, mps


if __name__ == '__main__':
    download()
