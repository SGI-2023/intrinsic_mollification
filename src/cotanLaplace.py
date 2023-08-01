'''
    Phase II:
    Build the cotan-Laplace matrix from the intrinsic lengths.
    (A description of how to do this can be found in the same paper by Sharp & Crane).
'''

import numpy as np
import igl
import Mollification

'''
    Parameters:
        V: List of coordinates vertices, a 1D array of length N
        F: List of faces, a N x 3 matrix where each entry is a indexed vertex
        delta: such that l_ij + l_jk > l_ki + delta
    Returns:
        Laplacian: the cotan-Laplace matrix of size #E x #E after applying intrinsic mollification
'''
def cotanLaplace(V, F, delta = 10e-4):
    E, newL, eps = Mollification.IntrinsicMollification(V, F, delta)
    Laplacian = np.zeros(E.shape[0], E.shape[0])

    print("E = ", E)
    print("Laplacian = ", Laplacian)

    # # Build the cotan-Laplace matrix from the intrinsic lengths
    # for fr in F:
    #     for i in range(3):
    #         # get the edge index
    #         e = E.index([fr[i], fr[(i+1)%3]])
    #         # get the edge length
    #         l = newL[e]
    #         # get the other two vertices
    #         v1 = fr[i]
    #         v2 = fr[(i+1)%3]
    #         v3 = fr[(i+2)%3]
    #         # get the cotan of the angle
    #         cot = igl.cotmatrix_entries(V, F)[e]
    #         # update the cotan-Laplace matrix
    #         Laplacian[v1, v2] += cot/l
    #         Laplacian[v2, v1] += cot/l
    #         Laplacian[v1, v1] -= cot/l
    #         Laplacian[v2, v2] -= cot/l



