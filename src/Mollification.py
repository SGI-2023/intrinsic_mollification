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

###############################         TESTS           ###############################           

####### Test above function
import trimesh

annulus = trimesh.creation.annulus(r_min = 1, r_max = 3, height = 5)
V = np.array(annulus.vertices)
F = np.array(annulus.faces)

delta = 0.2
E, newL, eps = IntrinsicMollification(V, F, delta) 

print("delta = ", delta)
print("epsilon = ", eps)
print("Original lengths = ", igl.edge_lengths(V, E)[:7])
print("New lengths = ", newL[:7])

## Testing from spot with degeneracies
import os
root_folder = os.getcwd()

delta = 1e-4
V2, F2 = igl.read_triangle_mesh(os.path.join(
            root_folder, "../data", "spot-degenerate.obj"))
E2, newL2, eps2 = IntrinsicMollification(V2, F2, delta)

print("delta = ", delta)
print("epsilon = ", eps2)
print("Original lengths = ", igl.edge_lengths(V2, E2)[:7])
print("New lengths = ", newL2[:7])