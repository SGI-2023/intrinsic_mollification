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

    Laplacian = np.zeros((E.shape[0], E.shape[0]))

    # sort edge entries so that E[:,0] < E[:,1]
    E = np.sort(E, axis = 1)

    # Build the cotan-Laplace matrix from the intrinsic lengths
    for fr in F:
        # for each face, get the 3 edge indices
        e1 = np.where((E[:,0] == min(fr[i], fr[(i+1)%3])) & (E[:,1] == max(fr[i], fr[(i+1)%3])))[0][0]
        e2 = np.where((E[:,0] == min(fr[(i+1)%3], fr[(i+2)%3])) & (E[:,1] == max(fr[(i+1)%3], fr[(i+2)%3])))[0][0]
        e3 = np.where((E[:,0] == min(fr[(i+2)%3], fr[i])) & (E[:,1] == max(fr[(i+2)%3], fr[i])))[0][0]

        l1 = newL[e1]
        l2 = newL[e2]
        l3 = newL[e3]

        s = (l1 + l2 + l3)/2
        area = np.sqrt(s*(s-l1)*(s-l2)*(s-l3))

        # get the cotan of the angle
        faceL = [l1, l2, l3]
        for i in range(3):
            # get w for corner

            # update the cotan-Laplace matrix

