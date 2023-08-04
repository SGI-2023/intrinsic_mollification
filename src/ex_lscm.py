import igl

import numpy as np

from scipy.sparse.linalg import eigs

from massmatrix import massmatrix
from Mollification import IntrinsicMollification
from cotanLaplace import cotanLaplace
from scipy.sparse import coo_matrix, bmat
from massmatrix import massmatrix

def vector_area_matrix(F):
    # Number of vertices
    n = F.max() + 1

    # Find boundary facets
    E = igl.boundary_facets(F)
    # Prepare lists to store rows, cols, and values for coo_matrix
    rows = []
    cols = []
    values = []

    for k in range(len(E)):
        i = E[k, 0]
        j = E[k, 1]
        rows.extend([i + n, j, i, j + n])
        cols.extend([j, i + n, j + n, i])
        values.extend([-0.25, -0.25, 0.25, 0.25])

    # Create coo_matrix from the triplets
    A = coo_matrix((values, (rows, cols)), shape=(n * 2, n * 2))

    return A

def repdiag(A, d):

    blocks = []
    for i in range(d):
        block = [None] * d
        block[i] = coo_matrix(A) 
        blocks.append(block)

    B = bmat(blocks)

    return B

def lscm_hessian(V, F):
    newL = IntrinsicMollification(V, F)[-1]
    # Assemble the area matrix (note that A is #Vx2 by #Vx2)
    A = vector_area_matrix(F)
    # Assemble the cotan laplacian matrix
    L = cotanLaplace(F, newL)
    L_flat = repdiag(L, 2)
    Q = -L_flat - 2. * A

    return Q

def scp(V, F):
    newL = IntrinsicMollification(V, F)[-1]
    Q = lscm_hessian(V, F)
    u,v = eigs(Q)
    sortedIndices = np.argsort(u)
    secondSmallEigVal = u[sortedIndices[1]]
    secondSmallEigVec = v[:,sortedIndices[1]]

    v_uv = secondSmallEigVec.reshape(-1, 2)
    return secondSmallEigVec, secondSmallEigVal, v_uv

# Example usage:
if __name__ == "__main__":
    # Assuming you have V and F defined as numpy arrays
    [V, F] = igl.read_triangle_mesh("../data/bunny.obj")
    scp(V, F)