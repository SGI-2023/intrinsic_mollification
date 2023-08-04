import igl
import trimesh

import numpy as np
import pyvista as pv
import meshplot as mp

from tqdm import tqdm
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, kron, diags

from cotanLaplace import cotanLaplace
from Mollification import IntrinsicMollification

    
[v, f] = igl.read_triangle_mesh("../data/bunny.obj")

[E, eps, newL] = IntrinsicMollification(v, f)

l = cotanLaplace(f, E)
vs = [v]


#Laplacian Smoothing
print('[*] Laplacian smoothing...')
for i in tqdm(range(10)):
    m = igl.massmatrix(v,f,igl.MASSMATRIX_TYPE_BARYCENTRIC)
    v = spsolve(m - 0.001 * l, m.dot(v))
    vs.append(v)

mp.plot(vs[0], f)
mp.plot(vs[4], f)
mp.plot(vs[-1], f)


# bound= np.array([2,1])
# boundary = igl.boundary_loop(f)
# bound[0] = boundary[0]
# bound[1] = boundary[int(boundary.size /2)]

# bc = np.array([0.0, 0.0], [1,0, 0.0])

# LSCM parametrization
# _, uv = igl.lscm(v, f, bound, bc)

# Least Squares Conformal Maps
# v and f are numpy arrays containing vertex and face information
# Create a 2x2 block diagonal matrix with L as the block
# igl.lscm()
# l_flat = kron(csr_matrix(np.eye(2)), l)
# # Compute the vector area matrix A
# # Note: This assumes the faces are oriented consistently (e.g., counterclockwise) for each triangle.
# # If the orientation is not consistent, the result might be incorrect.
# f_areas = np.linalg.norm(np.cross(v[f[:, 1], :] - v[f[:, 0], :], v[f[:, 2], :] - v[f[:, 0], :]) / 2.0, axis=1)
# A = diags(f_areas, 0).tocsr()
# print(A.shape, l_flat.shape)
# exit()
# e_matrix = l_flat - 2*A

# print('[*] LSCM...')
# print(e_matrix)
# Now, L_flat and A contain the equivalent of the C++ SparseMatrix<double> L_flat; and SparseMatrix<double> A; respectively.

# def lscm(v, f, bound, bc):
#     pass
