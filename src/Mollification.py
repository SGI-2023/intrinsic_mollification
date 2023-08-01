'''
    Phase I:
    Implement the intrinsic mollification scheme by
    (i) reading off the original edge lengths and
    (ii) implementing the formula defined by Sharp & Crane to get new lengths that describe higher-quality triangles.

'''

import igl

'''
    Parameters:
        V: List of coordinates vertices, a 1D array of length N
        F: List of faces, a N x 3 matrix where each entry is a indexed vertex
        delta: such that l_ij + l_jk > l_ki + delta

    Returns:
        L: columns correspond to original edges lengths [1,2],[2,0],[0,1]
        eps = max_T max(0, delta - l_ij - l_jk - l_ki )
        newL: new edges lengths in the same shape as L      
'''

def IntrinsicMollification(V, F, delta = 1e-4):
    L = igl.edge_lengths(V, F)         # columns correspond to edges lengths [1,2],[2,0],[0,1]
    eps = 0.0

    # iterate over each triangle to compute epsilon
    for T in L:
        eps = max(   [eps, delta + T[0] - T[1] - T[2], delta - T[0] + T[1] - T[2], delta - T[0] - T[1] + T[2] ]  )

    newL = eps + L
    return L, eps, newL