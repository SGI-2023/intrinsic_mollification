'''
    Phase I:
    Implement the intrinsic mollification scheme by
    (i) reading off the original edge lengths and
    (ii) implementing the formula defined by Sharp & Crane to get new lengths that describe higher-quality triangles.

'''

import numpy as np
import igl

'''
    Parameters:
        V: List of coordinates vertices, a 1D array of length N
        F: List of faces, a N x 3 matrix where each entry is a indexed vertex
        delta: such that l_ij + l_jk > l_ki + delta

    Returns:
        E: #E x 2 list of edges in no particular order
        newL: array of length #E with new lengths corresponding to the same order of E
        eps = max_T max(0, delta - l_ij - l_jk - l_ki )
'''
def IntrinsicMollification(V, F, delta = 10e-4):
    L = igl.edge_lengths(V, F)         # columns correspond to edges lengths [1,2],[2,0],[0,1]
    E = igl.edges(F)                   # #E x 2 list of edges in no particular order

    eps = 0.0
    # iterate over each triangle to compute epsilon
    for T in L:
        eps = max(   [0, eps, delta + T[0] - T[1] - T[2], delta - T[0] + T[1] - T[2], delta - T[0] - T[1] + T[2] ]  )

    newL = eps*np.ones(len(E))
    newL = newL + igl.edge_lengths(V, E)

    return E, newL, eps

