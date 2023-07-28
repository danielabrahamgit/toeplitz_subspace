from toeplitz_subspace.torch import SubspaceLinopFactory
from invivo_data import load_data

def test_forward():
    linop_factory = SubspaceLinopFactory(
        trj=data.trj,
        phi=data.phi,
        mps=data.mps,
        sqrt_dcf=data.sqrt_dcf,
        subsamp_idx=subsamp_idx
    )
    A, _, _ = linop_factory.get_forward()
    b = A(x)


if __name__ == '__main__':
    test_forward()
