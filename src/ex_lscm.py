import igl

import numpy as np

from scipy.sparse.linalg import eigs, spsolve, eigsh

from massmatrix import *
from Mollification import IntrinsicMollificationConstant
from cotanLaplace import *
from scipy.sparse import coo_matrix, bmat, csr_matrix

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

def lscm_hessian(V, F,
                 mollified=False,
                 neg_hack=NEG_HACK.NONE,
                 nan_hack=NAN_HACK.NONE,
                 close_zero_hack=CLOSE_TO_ZERO_HACK.NONE):

    edgeL = []
    if(mollified):
        edgeL = IntrinsicMollificationConstant(V, F, delta=1e1)[2]
    else:
        edgeL = igl.edge_lengths(V, F)
    #newL = IntrinsicMollificationConstant(V, F)[-1]
    #edgeL = igl.edge_lengths(V, F)

    # Assemble the area matrix (note that A is #Vx2 by #Vx2)
    A = vector_area_matrix(F)
    # Assemble the cotan laplacian matrix
    L = cotanLaplace(F, edgeL, neg_hack, nan_hack, close_zero_hack)
    L_flat = repdiag(L, 2)
    Q = -L_flat - 2. * A

    return Q

def lscm(V, F,
         mollified=False,
         neg_hack=NEG_HACK.NONE,
         nan_hack=NAN_HACK.NONE,
         close_zero_hack=CLOSE_TO_ZERO_HACK.NONE):

    # Fix the two points from the boundary.
    b = np.array([2, 1])
    bnd = igl.boundary_loop(F)
    b[0] = bnd[0]
    b[1] = bnd[int(bnd.size / 2)]

    bc = np.array([[0.0, 0.0], [1.0, 0.0]])

    Q = lscm_hessian(V, F, mollified, neg_hack, nan_hack, close_zero_hack)

    # Min quad parameters.
    b_flat = np.zeros([b.size * np.shape(bc)[1]], dtype=int)
    bc_flat = np.zeros([bc.size], dtype=float)
    B_flat = np.zeros([2*np.shape(V)[0]], dtype=float)
    Aeq = csr_matrix(np.shape(Q))
    Beq = np.zeros([np.shape(Q)[0], 1], dtype=float)

    # Sizes.
    V_rows = np.shape(V)[0]
    bc_rows = np.shape(bc)[0]
    bc_cols = np.shape(bc)[1]
    b_size = np.shape(b)[0]

    # Initial values for flat b and bc.
    for c in range(bc_cols):
        b_flat[c*b_size:(c*b_size + 2)] = c*V_rows + b
        bc_flat[c*bc_rows:((c+1)*bc_rows)] = bc[:, c]

    # IGL performs precompute and solve for min_quad_with fixed in C++.
    # However, only the direct function min_quad_with_fixed is available
    # in Python.
    done, W_flat = igl.min_quad_with_fixed(Q, B_flat, b_flat, bc_flat, Aeq, Beq, True)

    V_uv = np.zeros([V_rows, 2])
    V_uv_rows = np.shape(V_uv)[0]
    V_uv_cols = np.shape(V_uv)[1]

    # Results from [x, x, ..., y, y] to [x, y, ..., x, y].
    for i in range(V_uv_cols):
        V_uv[:, i] = W_flat[(i*V_uv_rows):((i+1)*V_uv_rows)]

    return done, V_uv

def scp(V, F):
    Q = lscm_hessian(V, F)
    u,v = eigs(Q, k=2, which='SR')
    secondSmallEigVal = u[1]
    secondSmallEigVec = v[:,1]

    # v_uv = secondSmallEigVec.reshape(-1, 2)
    vec_len = len(secondSmallEigVec)
    x_vecs = secondSmallEigVec[:int(vec_len//2)].reshape(-1, 1)
    y_vecs = secondSmallEigVec[int(vec_len//2):].reshape(-1, 1)
    v_uv = np.hstack((x_vecs, y_vecs))

    return secondSmallEigVec, secondSmallEigVal, v_uv

def isometric_distortion(V, F, V_uv):
    # compute the edge lengths of the original mesh
    L = igl.edge_lengths(V, F)
    # compute the edge lengths of the parametrized mesh
    if V_uv.shape[1] == 2:
        V_uv = np.hstack([V_uv, np.zeros([V_uv.shape[0], 1])])
    L_uv = igl.edge_lengths(V_uv, F)

    # normalize the edge lengths
    L = L / np.linalg.norm(L)
    L_uv = L_uv / np.linalg.norm(L_uv)

    # compute the isometric distortion
    isometric_distortion = np.linalg.norm(L - L_uv)

    return isometric_distortion

def area_distortion(V, F, V_uv):
    # compute the area of the original mesh
    A = igl.doublearea(V, F)
    # compute the area of the parametrized mesh
    A_uv = igl.doublearea(V_uv, F)

    # normalize the areas
    A = A / np.linalg.norm(A)
    A_uv = A_uv / np.linalg.norm(A_uv)

    # compute the area distortion
    area_distortion = np.linalg.norm(A - A_uv)

    return area_distortion


# We are using the quasi conformal error as the metric for evaluation
# of the conformal mapping algorithms like LSCM and SCP


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
    e1 = u1 / (np.linalg.norm(u1) + 1e-20)
    e2 = u2 - np.dot(u2, e1) * e1
    e2 = e2 / (np.linalg.norm(e2) + 1e-20)

    # get the orthonormal basis of the face in uv space
    f1 = v1 / (np.linalg.norm(v1) + 1e-20)
    f2 = v2 - np.dot(v2, f1) * f1
    f2 = f2 / (np.linalg.norm(f2) + 1e-20)

    # project the edge vectors of the face in R3 to the orthonormal basis
    # of the face in R3
    p = np.zeros([3, 3])
    q = np.zeros([3, 3])

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
    det = np.sqrt(max(np.power(a - c, 2) + 4.0 * b * b, 0))
    gamma_max = np.sqrt(max(0.5 * (a + c + det), 0))
    gamma_min = np.sqrt(max(0.5 * (a + c - det), 0))

    if gamma_max < gamma_min:
        gamma_max, gamma_min = gamma_min, gamma_max

    return gamma_max / max(gamma_min, 1e-20), area_p

def lscm_hessian_L(F, newL):
    #newL = IntrinsicMollificationConstant(V, F)[-1]
    #edgeL = igl.edge_lengths(V, F)
    # Assemble the area matrix (note that A is #Vx2 by #Vx2)
    A = vector_area_matrix(F)
    # Assemble the cotan laplacian matrix
    L = cotanLaplace(F, newL)
    L_flat = repdiag(L, 2)
    Q = -L_flat - 2. * A

    return Q

def lscm_L(V, F, newL):
    # Fix the two points from the boundary.
    b = np.array([2, 1])
    bnd = igl.boundary_loop(F)
    b[0] = bnd[0]
    b[1] = bnd[int(bnd.size / 2)]

    bc = np.array([[0.0, 0.0], [1.0, 0.0]])

    Q = lscm_hessian_L(F, newL)

    # Min quad parameters.
    b_flat = np.zeros([b.size * np.shape(bc)[1]], dtype=int)
    bc_flat = np.zeros([bc.size], dtype=float)
    B_flat = np.zeros([2*np.shape(V)[0]], dtype=float)
    Aeq = csr_matrix(np.shape(Q))
    Beq = np.zeros([np.shape(Q)[0], 1], dtype=float)

    # Sizes.
    V_rows = np.shape(V)[0]
    bc_rows = np.shape(bc)[0]
    bc_cols = np.shape(bc)[1]
    b_size = np.shape(b)[0]

    # Initial values for flat b and bc.
    for c in range(bc_cols):
        b_flat[c*b_size:(c*b_size + 2)] = c*V_rows + b
        bc_flat[c*bc_rows:((c+1)*bc_rows)] = bc[:, c]

    # IGL performs precompute and solve for min_quad_with fixed in C++.
    # However, only the direct function min_quad_with_fixed is available
    # in Python.
    done, W_flat = igl.min_quad_with_fixed(Q, B_flat, b_flat, bc_flat, Aeq, Beq, True)

    V_uv = np.zeros([V_rows, 2])
    V_uv_rows = np.shape(V_uv)[0]
    V_uv_cols = np.shape(V_uv)[1]

    # Results from [x, x, ..., y, y] to [x, y, ..., x, y].
    for i in range(V_uv_cols):
        V_uv[:, i] = W_flat[(i*V_uv_rows):((i+1)*V_uv_rows)]

    return done, V_uv

def scp_L(F, FL):
    Q = lscm_hessian_L(F, FL)
    avg_edge_length = np.mean(FL)
    # use shift-invert mode to find the 2 smallest eigenvalues
    u,v = eigs(Q, k=2, which='LM', tol=1e-2, sigma=0.0001)
    # replace eigs with manual power iteration
    # v = power_iteration(Q)
    #secondSmallEigVal = u[1]
    secondSmallEigVal = u[1]
    secondSmallEigVec = v[:,1]

    # v_uv = secondSmallEigVec.reshape(-1, 2)
    vec_len = len(secondSmallEigVec)
    x_vecs = secondSmallEigVec[:int(vec_len//2)].reshape(-1, 1)
    y_vecs = secondSmallEigVec[int(vec_len//2):].reshape(-1, 1)
    v_uv = np.hstack((x_vecs, y_vecs))
    v_uv = np.real(v_uv)

    return secondSmallEigVec, secondSmallEigVal, v_uv

def harmonic_L(V, F, FL):
    bnd = igl.boundary_loop(F)

    ## Map the boundary to a circle, preserving edge proportions
    bnd_uv = igl.map_vertices_to_circle(V, bnd)

    L = cotanLaplace(F, FL)
    M = massmatrix(FL, F, MASSMATRIX_TYPE.BARYCENTRIC)
    # M is scipy.sparse._dia.dia_matrix, L is scipy.sparse._lil.lil_matrix
    # we need to convert them to scipy.sparse.csr.csr_matrix
    M = M.tocsr()
    L = L.tocsr()

    uv = igl.harmonic_weights_from_laplacian_and_mass(L, M, bnd, bnd_uv, 1)

    V_uv = np.hstack([uv, np.zeros((uv.shape[0],1))])

    return V_uv


# should return the second smallest eigenvalue and eigenvector
def power_iteration(A, num_simulations=50):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k2 = np.random.rand(A.shape[0])
    for _ in range(num_simulations):
        # solve
        b_k2 = spsolve(A, b_k2)
        # substract mean
        b_k2 = b_k2 - np.mean(b_k2)

        # normalize
        b_k2 = b_k2 / np.linalg.norm(b_k2)
    return b_k2

#         if risidual(A, b_k2) < epsillon:
#             break

# def risidual(A, b):
#     # in c++:
#     # Vector<std::complex<double>> bConj = b.conjugate().transpose();
#     # double lambda = (bConj * A * b).norm() / (bConj * b).norm();
#     # double res = (A * b - lambda * b).norm() / b.norm();

#     # in python:
#     # first, b is a real vector having the imaginary part in the second half
#     b_conj = b
#     b_conj[int(len(b)/2):] = -b_conj[int(len(b)/2):]

#     lambda_ = np.linalg.norm(np.dot(np.dot(b_conj, A), b)) / np.linalg.norm(np.dot(b_conj, b))
#     res = np.linalg.norm(np.dot(A, b) - lambda_ * b) / np.linalg.norm(b)

#     return res
