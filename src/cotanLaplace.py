'''
    Phase II:
    Build the cotan-Laplace matrix from the intrinsic lengths.
    (A description of how to do this can be found in the same paper by Sharp & Crane).
'''
import scipy as sp
import numpy as np
import igl
from enum import Enum


class NEG_HACK(Enum):
    NONE=0
    TO_ABS=1
    TO_ONE=2

class NAN_HACK(Enum):
    NONE=0
    TO_ZERO=1
    TO_ONE=2

class CLOSE_TO_ZERO_HACK(Enum):
    NONE=0
    TO_ZERO=1


'''
    Parameters:
        F: List of faces, a N x 3 matrix where each entry is a indexed vertex
        L: Lengths of edges in the order of edges [1, 2], [2, 0], [0, 1].
        neg_hack: How to handle negative weights.   Options: "none" (default), "to_abs", "to_one".
        nan_hack: How to handle NaN weights.        Options: "none" (default), "to_zero", "to_one".
        close_zero_hack: How to handle close to zero weights. Options: "none" (default), "to_zero". 
    Returns:
        Laplacian: Laplacian in format of a lil_matrix from scipy sparse matrix.
'''


def cotanLaplace(F, L, 
                 neg_hack=NEG_HACK.NONE, 
                 nan_hack=NAN_HACK.NONE,
                 close_zero_hack=CLOSE_TO_ZERO_HACK.NONE):

    Vsize = np.max(F) + 1

    Laplacian = sp.sparse.lil_matrix((Vsize, Vsize))

    for face in range(len(F)):
        edges_length = L[face]
        e0 = edges_length[0]
        e1 = edges_length[1]
        e2 = edges_length[2]

        s = (e0 + e1 + e2)/2
        area2 = s*(s-e1)*(s-e2)*(s-e0)
        if area2 < 0:
            area2 = 0
        area = np.sqrt(area2)

        lengths = [e0, e1, e2]
        for i in range(3):  # Consider face ijk
            j = (i+1) % 3
            k = (i+2) % 3
            # Get w per corner:
            wjk = (lengths[j] ** 2 + lengths[k] ** 2 - lengths[i] ** 2) / (8*area)

            # Handle negative weights.
            if  wjk < 0 and neg_hack != NEG_HACK.NONE:
                if neg_hack == NEG_HACK.TO_ABS:
                    wjk = - wjk
                elif neg_hack == NEG_HACK.TO_ONE:
                    wjk = 1

            # Handle NaN weights.
            if (np.isnan(wjk) or np.isinf(wjk)) and nan_hack != NAN_HACK.NONE:
                if nan_hack == NAN_HACK.TO_ZERO:
                    wjk = 0
                elif nan_hack == NAN_HACK.TO_ONE:
                    wjk = 1

            # Handle close to zero weights.
            if(np.abs(wjk) < 1e-12 and close_zero_hack != CLOSE_TO_ZERO_HACK.NONE):
                wjk = 0

            # Update the cotan-Laplace matrix.
            fr = F[face]
            Laplacian[fr[j], fr[j]] -= wjk  # L_jj
            Laplacian[fr[k], fr[k]] -= wjk  # L_kk
            Laplacian[fr[j], fr[k]] += wjk  # L_jk
            Laplacian[fr[k], fr[j]] += wjk  # L_kj

    return Laplacian
