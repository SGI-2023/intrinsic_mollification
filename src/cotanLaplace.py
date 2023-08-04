'''
    Phase II:
    Build the cotan-Laplace matrix from the intrinsic lengths.
    (A description of how to do this can be found in the same paper by Sharp & Crane).
'''
import scipy as sp
import numpy as np
import igl

'''
    Parameters:
        F: List of faces, a N x 3 matrix where each entry is a indexed vertex
        L: Lengths of edges in the order of edges [1, 2], [2, 0], [0, 1].
    Returns:
        Laplacian: Laplacian in format of a lil_matrix from scipy sparse matrix.
'''


def cotanLaplace(F, L, neg_hack="none", nan_hack="none"):

    Vsize = np.max(F) + 1

    Laplacian = sp.sparse.lil_matrix((Vsize, Vsize))

    for face in range(len(F)):
        edges_length = L[face]
        e0 = edges_length[0]
        e1 = edges_length[1]
        e2 = edges_length[2]

        s = (e0 + e1 + e2)/2
        area = np.sqrt(s*(s-e1)*(s-e2)*(s-e0))

        lengths = [e0, e1, e2]
        for i in range(3):  # Consider face ijk
            j = (i+1) % 3
            k = (i+2) % 3
            # Get w per corner:
            wjk = (lengths[j] ** 2 + lengths[k] ** 2 - lengths[i] ** 2) / (8*area)

            if  wjk < 0 and neg_hack != "none":
                if neg_hack == "to_abs":
                    wjk = - wjk
                elif neg_hack == "to_one":
                    wjk = 1

            if np.isnan(wjk) and nan_hack != "none":
                if nan_hack == "to_zero":
                    wjk = 0
                elif nan_hack == "to_one":
                    wjk = 1

            # Update the cotan-Laplace matrix.
            fr = F[face]
            Laplacian[fr[j], fr[j]] -= wjk  # L_jj
            Laplacian[fr[k], fr[k]] -= wjk  # L_kk
            Laplacian[fr[j], fr[k]] += wjk  # L_jk
            Laplacian[fr[k], fr[j]] += wjk  # L_kj

    return Laplacian
