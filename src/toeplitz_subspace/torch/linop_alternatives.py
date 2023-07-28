"""
Collection of linop functions that aren't being used at the moment
"""
import torch

def A_func_mem_inefficient(x: torch.Tensor):
    """
    Don't use - regular method is more memory-efficient
    x: [A *im_size]
    """
    assert x.shape == self.ishape, f'Shape mismatch: x: {x.shape}, expected {self.ishape}'
    # Apply subspace basis
    x = torch.einsum('at,a...->t...', self.phi, x[:, None, ...]) # [T C *im_size]
    y = scale_factor * self.nufft(x, self.trj, smaps=self.mps, norm=norm)  # [T C K]
    y = self.sqrt_dcf[:, None, :] * y # [T C K]
    # Subsample
    return y # [T C K]

def A_func_mem_inefficient(x: torch.Tensor):
    """
    x: [A *im_size]
    """
    assert x.shape == self.ishape, f'Shape mismatch: x: {x.shape}, expected {self.ishape}'
    # Apply subspace basis
    x = torch.einsum('at,a...->t...', self.phi, x[:, None, ...]) # [T C *im_size]
    y = torch.zeros((T, C, K), dtype=torch.complex64, device=x.device)
    for c, d in batch_iterator(C, coil_batch):
        y[:, c:d, :] = scale_factor * self.nufft(x,
                                                    self.trj,
                                                    smaps=self.mps[c:d],
                                                    norm=norm)  # [T C K]
    y = self.sqrt_dcf[:, None, :] * y # [T C K]
    # Subsample
    return y # [T C K]

def AH_func_mem_inefficient(y: torch.Tensor):
    assert y.shape == self.oshape, f'Shape mismatch: y: {y.shape}, expected {self.oshape}'
    # y_out = torch.zeros((T, C, K), device=y.device).type(y.dtype)
    # Apply adjoint density compensation
    y = self.sqrt_dcf[:, None, :] * y
    # Apply Adjoint NUFFT and coils
    x = scale_factor * self.nufft_adjoint(y, self.trj, smaps=self.mps, norm=norm) # [T H W]
    # Remove leftover coil dim
    x = x[:, 0, ...]
    orig_xshape = x.shape[1:]
    # Apply adjoint subspace
    x = rearrange(x, 't ... -> t (...)')
    x = torch.einsum('at,td->ad', torch.conj(self.phi), x)
    x = x.reshape(x.shape[0], *orig_xshape)
    return x

def AH_func(y: torch.Tensor):
    assert y.shape == self.oshape, f'Shape mismatch: y: {y.shape}, expected {self.oshape}'
    # y_out = torch.zeros((T, C, K), device=y.device).type(y.dtype)
    # Apply adjoint density compensation
    y = self.sqrt_dcf[:, None, :] * y
    # Apply Adjoint NUFFT and coils
    # Parallelize across coils to save memory
    x = 0
    orig_device = y.device
    for c, d in tqdm(batch_iterator(C, coil_batch),
                        total=C//coil_batch,
                        desc='Coils'):
        x_coil = 0
        for e, f in tqdm(batch_iterator(R, trj_batch),
                            total=R//trj_batch,
                            leave=False,
                            desc='Trjs'):
            x_tmp = scale_factor * self.nufft_adjoint(y[e:f, c:d].to(nufft_device),
                                                    self.trj[e:f].to(nufft_device),
                                                    smaps=self.mps[c:d].to(nufft_device), norm=norm) # [T H W]
            # Apply adjoint subspace (in inner loop to save memory)
            x_tmp = torch.einsum('at,t...->a...', torch.conj(self.phi[:, e:f].to(nufft_device)), x_tmp)
            x_coil += x_tmp.to(orig_device)
        x += x_coil[:, 0, ...] # Remove leftover coil dim
    return x
