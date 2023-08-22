import igl

import numpy as np

from scipy.sparse.linalg import eigs

from massmatrix import massmatrix
from Mollification import IntrinsicMollificationConstant
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
    #newL = IntrinsicMollificationConstant(V, F)[-1]
    edgeL = igl.edge_lengths(V, F)
    # Assemble the area matrix (note that A is #Vx2 by #Vx2)
    A = vector_area_matrix(F)
    # Assemble the cotan laplacian matrix
    L = cotanLaplace(F, edgeL)
    L_flat = repdiag(L, 2)
    Q = -L_flat - 2. * A

    return Q

def scp(V, F):
    newL = IntrinsicMollificationConstant(V, F)[-1]
    Q = lscm_hessian(V, F)
    u,v = eigs(Q)
    sortedIndices = np.argsort(u)
    secondSmallEigVal = u[sortedIndices[1]]
    secondSmallEigVec = v[:,sortedIndices[1]]

    v_uv = secondSmallEigVec.reshape(-1, 2)
    return secondSmallEigVec, secondSmallEigVal, v_uv

# quasi conformal error for evaluation
# V: vertex positions (#V x 3)
# F: faces defined by vertex indices (#F x 3)
# uv: the eigen vector reshaped to (#V x 2)
def quasi_conformal_error(V, F, uv):
    # Keep track of the total, max and min error
    total_qc = 0
    max_qc = -np.inf
    min_qc = np.inf
    total_area = 0

    for f in F:
        # get the vertex positions of the face as a 3x3 matrix
        p = V[f]
        # get the uv coordinates of the face as a 3x2 matrix
        q = uv[f]

        qc, area = quasi_conformal_error_per_face(p, q)

        max_qc = max(max_qc, qc)
        min_qc = min(min_qc, qc)

        total_qc += qc * area
        total_area += area

    return total_qc / total_area, max_qc, min_qc

# quasi conformal error per face
# p: vertex positions of the face as a 3x3 matrix
# q: uv coordinates of the face as a 3x2 matrix
def quasi_conformal_error_per_face(p, q):
    # compute the edge vectors of the face in R3
    u1 = p[1] - p[0]
    u2 = p[2] - p[0]

    # compute the edge vectors of the face in uv space
    v1 = q[1] - q[0]
    v2 = q[2] - q[0]

    # get the orthonormal basis of the face in R3
    e1 = u1 / np.linalg.norm(u1)
    e2 = u2 - np.dot(u2, e1) * e1
    e2 = e2 / np.linalg.norm(e2)

    # get the orthonormal basis of the face in uv space
    f1 = v1 / np.linalg.norm(v1)
    f2 = v2 - np.dot(v2, f1) * f1
    f2 = f2 / np.linalg.norm(f2)

    # project the edge vectors of the face in R3 to the orthonormal basis
    # of the face in R3
    p[0] = np.array([0, 0, 0])
    p[1] = np.array([np.dot(u1, e1), np.dot(u1, e2), 0])
    p[2] = np.array([np.dot(u2, e1), np.dot(u2, e2), 0])

    q[0] = np.array([0, 0, 0])
    q[1] = np.array([np.dot(v1, f1), np.dot(v1, f2), 0])
    q[2] = np.array([np.dot(v2, f1), np.dot(v2, f2), 0])

    # compute the area of the face in R3
    area_p = np.linalg.norm(np.cross(p[1], p[2])) / 2

    # compute singular values of mapping from 3D to 2D
    Ss = (q[0] * (p[1][1] - p[2][1]) + q[1] * (p[2][1] - p[0][1]) + q[2] * (p[0][1] - p[1][1])) / area_p
    St = (q[0] * (p[2][0] - p[1][0]) + q[1] * (p[0][0] - p[2][0]) + q[2] * (p[1][0] - p[0][0])) / area_p
    a = np.dot(Ss, Ss)
    b = np.dot(Ss, St)
    c = np.dot(St, St)
    det = np.sqrt(np.power(a - c, 2) + 4.0 * b * b)
    gamma_max = np.sqrt(0.5 * (a + c + det))
    gamma_min = np.sqrt(0.5 * (a + c - det))

    if gamma_max < gamma_min:
        gamma_max, gamma_min = gamma_min, gamma_max

    return gamma_max / gamma_min, area_p








# Example usage:
if __name__ == "__main__":
    # Assuming you have V and F defined as numpy arrays
    [V, F] = igl.read_triangle_mesh("../data/bunny.obj")
    scp(V, F)