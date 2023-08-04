import igl

import numpy as np

from massmatrix import massmatrix
from Mollification import IntrinsicMollification
from cotanLaplace import cotanLaplace
from scipy.sparse import coo_matrix, bmat

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


def lscm(V, F):
    """
    This function is getting vertices, face, boundry and boundry condition and returns 
    :

    """
    b = igl.boundary_facets(F)
    bc = 
    Q = lscm_hessian(V, F)

    b_flat = np.repeat(b * V.shape[0], bc.shape[1])
    bc_flat = bc.flatten()

    B_flat = np.zeros(V.shape[0] * 2)

    data = igl.min_quad_with_fixed_precompute(Q, b_flat, igl.EigenSparseMatrixd(), True)
    W_flat = igl.min_quad_with_fixed_solve(data, B_flat, bc_flat)

    V_uv = np.zeros((V.shape[0], 2))
    V_uv[:, 0] = W_flat[:V.shape[0]]
    V_uv[:, 1] = W_flat[V.shape[0]:]

    return V_uv, Q


from scipy.sparse import coo_matrix

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




# Example usage:
if __name__ == "__main__":
    # Assuming you have V and F defined as numpy arrays
    [V, F] = igl.read_triangle_mesh("../data/bunny.obj")
    lscm(V, F)

    # F = np.array([[0, 1, 2], [1, 3, 2]])

    # A = vector_area_matrix(F)
    # print(A)


    # V = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    # F = np.array([[0, 1, 2], [1, 3, 2]])
    # b = np.array([0, 1])
    # bc = np.array([[0, 0], [1, 0]])

    # V_uv = lscm(V, F, b, bc)

    # F = np.array([[0, 1, 2], [1, 3, 2]])

    # A = vector_area_matrix(F)
    # print(A)