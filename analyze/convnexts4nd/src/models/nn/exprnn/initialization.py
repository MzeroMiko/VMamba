# Downloaded from https://github.com/Lezcano/expRNN

import torch
import numpy as np
import scipy.linalg as la


def henaff_init_(A):
    size = A.size(0) // 2
    diag = A.new(size).uniform_(-np.pi, np.pi)
    return create_diag_(A, diag)


def cayley_init_(A):
    size = A.size(0) // 2
    diag = A.new(size).uniform_(0., np.pi / 2.)
    diag = -torch.sqrt((1. - torch.cos(diag))/(1. + torch.cos(diag)))
    return create_diag_(A, diag)

# We include a few more initializations that could be useful for other problems
def haar_init_(A):
    """Haar initialization on SO(n)."""
    torch.nn.init.orthogonal_(A)
    with torch.no_grad():
        if A.det() < 0.:
            # Go bijectively from O^-(n) to O^+(n) \iso SO(n)
            idx = np.random.randint(0, A.size(0))
            A[idx] *= -1.
        An = la.logm(A.data.cpu().numpy()).real
        An = .5 * (An - An.T)
        A.copy_(torch.tensor(An))
        return A


def haar_diag_init_(A):
    """Block-diagonal skew-symmetric matrix with eigenvalues distributed as those from a Haar."""
    haar_init_(A)
    with torch.no_grad():
        An = A.data.cpu().numpy()
        eig = la.eigvals(An).imag
        eig = eig[::2]
        if A.size(0) % 2 == 1:
            eig = eig[:-1]
        eig = torch.tensor(eig)
        return create_diag_(A, eig)


def normal_squeeze_diag_init_(A):
    size = A.size(0) // 2
    diag = A.new(size).normal_(0, 1).fmod_(np.pi/8.)
    return create_diag_(A, diag)

def normal_diag_init_(A):
    size = A.size(0) // 2
    diag = A.new(size).normal_(0, 1).fmod_(np.pi)
    return create_diag_(A, diag)


def create_diag_(A, diag):
    n = A.size(0)
    diag_z = torch.zeros(n-1)
    diag_z[::2] = diag
    A_init = torch.diag(diag_z, diagonal=1)
    A_init = A_init - A_init.T
    with torch.no_grad():
        A.copy_(A_init)
        return A
