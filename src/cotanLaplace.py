'''
    Phase II:
    Build the cotan-Laplace matrix from the intrinsic lengths.
    (A description of how to do this can be found in the same paper by Sharp & Crane).
'''

import numpy as np
import igl

'''
    Parameters:
        V: List of coordinates vertices, a 1D array of length N
        F: List of faces, a N x 3 matrix where each entry is a indexed vertex
        delta: such that l_ij + l_jk > l_ki + delta
    Returns:
        Laplacian: the cotan-Laplace matrix of size #E x #E after applying intrinsic mollification
'''
def cotanLaplace(F, E, L, Vsize):

    LaplacianMollified = np.zeros((Vsize, Vsize))

    # sort edge entries so that E[:,0] < E[:,1]
    E = np.sort(E, axis = 1)

    # Build the cotan-Laplace matrix from the intrinsic lengths
    for fr in F:
        # for each face, get the 3 edge indices
        e0 = np.where((E[:,0] == min(fr[1], fr[2])) & (E[:,1] == max(fr[1], fr[2])))[0][0]
        e1 = np.where((E[:,0] == min(fr[2], fr[0])) & (E[:,1] == max(fr[2], fr[0])))[0][0]
        e2 = np.where((E[:,0] == min(fr[0], fr[1])) & (E[:,1] == max(fr[0], fr[1])))[0][0]

        l0 = L[e0]
        l1 = L[e1]
        l2 = L[e2]

        s = (l0 + l1 + l2)/2
        area = np.sqrt(s*(s-l1)*(s-l2)*(s-l0))

        # get the cotan of the angle
        faceL = [l0, l1, l2]
        for i in range(3):
            # get w for corner
            wjk = (faceL[(i+1)%3] ** 2 + faceL[(i+2)%3] ** 2 - faceL[i] ** 2) / (8 * area)

            # update the cotan-Laplace matrix
            LaplacianMollified[fr[(i+1)%3], fr[(i+1)%3]] -= wjk
            LaplacianMollified[fr[(i+2)%3], fr[(i+2)%3]] -= wjk
            LaplacianMollified[fr[(i+1)%3], fr[(i+2)%3]] += wjk
            LaplacianMollified[fr[(i+2)%3], fr[(i+1)%3]] += wjk

    return LaplacianMollified

