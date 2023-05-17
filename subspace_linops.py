import sigpy as sp
import numpy as np
import time

def subspace_linop(trj, phi, mps, sqrt_dcf=None):
    """
    Constructs a linear operator that describes the subspace reconstruction model.
    This is the memory efficient implimentation
        trj - trajectory with shape (nro, npe, ntr, d)
        phi - subspace basis with shape (nsub, ntr)
        mps - sensititvity maps with shape (ncoil, ndim1, ..., ndimN)
        sqrt_dcf - the density comp. functon with shape (nro, ...)
        use_gridding - toggles gridding reconstruction
    """

    dev = sp.get_device(mps)
    assert sp.get_device(phi) == dev

    F = sp.linop.NUFFT(mps.shape[1:], trj)
    if type(sqrt_dcf) != type(None):
        assert sp.get_device(sqrt_dcf) == dev
        F = sp.linop.Multiply(trj.shape[:-1], sqrt_dcf) * F

    outer_A = []
    for k in range(mps.shape[0]):
        S = sp.linop.Multiply(mps.shape[1:], mps[k, ...]) * \
            sp.linop.Reshape( mps.shape[1:], [1] + list(mps.shape[1:]))
        lst_A = [sp.linop.Reshape([1] + list(F.oshape), F.oshape)   * \
                    sp.linop.Multiply(F.oshape, phi[k, None, :]) * \
                    F * S for k in range(phi.shape[0])]
        inner_A = sp.linop.Hstack(lst_A, axis=0)
        D1 = sp.linop.ToDevice(inner_A.ishape, dev, sp.cpu_device)
        D2 = sp.linop.ToDevice(inner_A.oshape, sp.cpu_device, dev)
        outer_A.append(D2 * inner_A * D1) 
    A = sp.linop.Vstack(outer_A, axis=0)
        
    return A

class A_subspace(sp.linop.Linop):
    def __init__(self, trj, phi, mps, sqrt_dcf=None, fast_AHA=False, mem_save=True):
        super().__init__((mps.shape[0], *trj.shape[:-1]), (phi.shape[0], *mps.shape[1:]))

        # Save stuff
        self.dev = sp.get_device(mps)
        self.fast_AHA = fast_AHA
        
        # Define standard subspace linop
        self.linop = subspace_linop(trj, phi, mps, sqrt_dcf)
        
        # Store toeplitz embedding
        if fast_AHA == True:
            print(f'\nCalculating Toeplitz Embeddings')
            start = time.perf_counter()

            # Constants
            n_trj, n_pe, n_tr, d = trj.shape
            n_subspace = phi.shape[0]
            n_coil = mps.shape[0]
            im_size = mps.shape[1:]
            os_factor = 2
            im_size_os = list((np.array(im_size) * os_factor).astype(int))
            alpha_size = (n_subspace, *im_size)
            alpha_size_os = (n_subspace, *im_size_os)   

            # Calculate toeplitz embeddings
            Ts = np.zeros((n_subspace, n_subspace, *im_size_os), dtype=np.complex64)
            phi_cpu = sp.to_device(phi, sp.cpu_device)
            for k in range(n_subspace):
                print(f'k = {k+1}/{n_subspace}', end='\r', flush=True)
                
                # Set kth coeff to zero, which is all ones in freq domain
                alpha_ksp = np.zeros((n_subspace, *trj.shape[:-1]), dtype=np.complex64)
                alpha_ksp[k, ...] = 1
                
                # Transform with PHI
                phi_rs = np.reshape(phi_cpu, (n_subspace, 1, 1, n_tr))
                sig_ksp = np.sum(alpha_ksp * phi_rs, axis=0)
                
                # And go backwards
                alpha_ksp = sig_ksp[None, ...] * phi_rs.conj()
                
                # Adjoint on GPU
                with self.dev:
                    alpha_ksp = sp.to_device(alpha_ksp, self.dev)

                    if sqrt_dcf is None:
                        psf_col = sp.nufft_adjoint(input=alpha_ksp, 
                                                    coord=trj * os_factor,
                                                    oshape=alpha_size_os)
                    else:
                        psf_col = sp.nufft_adjoint(input=alpha_ksp * sqrt_dcf ** 2, 
                                                    coord=trj * os_factor,
                                                    oshape=alpha_size_os)
                    T_col = sp.fft(input=psf_col,
                                   oshape=alpha_size_os,
                                   axes=(list(range(-d, 0)))) * (os_factor ** d)
                    T_col = sp.to_device(T_col, sp.cpu_device)

                    # Clear memory
                    del alpha_ksp, psf_col
                    if 'cuda' in str(self.dev).lower():
                        mempool = self.dev.xp.get_default_memory_pool()
                        pinned_mempool = self.dev.xp.get_default_pinned_memory_pool()
                        mempool.free_all_blocks()
                        pinned_mempool.free_all_blocks()
                    
                # Update Ts
                Ts[:, k, ...] = T_col

            # Make Toeplitz operator
            D1 = sp.linop.ToDevice(alpha_size, self.dev, sp.cpu_device)
            D2 = sp.linop.ToDevice(alpha_size, sp.cpu_device, self.dev)
            R2X = sp.linop.Resize(alpha_size_os, alpha_size)
            F2X = sp.linop.FFT(alpha_size_os, axes=list(range(-d, 0)))
            R   = sp.linop.Reshape((1, *alpha_size_os), alpha_size_os)
            T   = sp.linop.Multiply(R.oshape, Ts)
            SUM = sp.linop.Sum(T.oshape, axes=(1,))
            D_presum = sp.linop.ToDevice(R.oshape, sp.cpu_device, self.dev)
            D_pstsum = sp.linop.ToDevice(SUM.oshape, self.dev, sp.cpu_device)
            outer_toep = []
            for k in range(n_coil):
                S = sp.linop.Multiply(alpha_size, mps[None, k, ...])
                op = D2 * S.H * R2X.H * F2X.H * SUM * T * R * F2X * R2X * S * D1
                R_coil = sp.linop.Reshape((1, *op.oshape), op.oshape)
                outer_toep.append(R_coil * op)
            outer_toep = sp.linop.Vstack(outer_toep, axis=0)
            self.toep_linop = sp.linop.Sum(outer_toep.oshape, axes=(0,)) * outer_toep
        
            end = time.perf_counter()
            print(f'Time = {end - start:.3f}s')

    def _apply(self, input):
        return self.linop * input

    def _adjoint_linop(self):
        return self.linop.H

    def _normal_linop(self):
        if self.fast_AHA:
            return self.toep_linop
        else:
            return self.linop.H * self.linop