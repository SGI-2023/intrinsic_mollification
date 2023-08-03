import trimesh
import os
import numpy as np
import igl
from Mollification import *
from cotanLaplace import *
from massmatrix import *



root_folder = os.getcwd()

###############################         TESTS           ###############################

####### Test IntrinsicMollification function
print("Test IntrinsicMollification function with annulus mesh")

annulus = trimesh.creation.annulus(r_min = 1, r_max = 3, height = 5)
V = np.array(annulus.vertices)
F = np.array(annulus.faces)

delta = 0.2
E, eps, newL = IntrinsicMollification(V, F, delta) 

print("delta = ", delta)
print("epsilon = ", eps)
print("Original lengths = ", igl.edge_lengths(V, F)[:7])
print("New lengths = ", newL[:7])

## Testing from spot with degeneracies
print("______________________________________________________________")
print("Test IntrinsicMollification function with spot-degenerate mesh")

delta = 1e-4
V2, F2 = igl.read_triangle_mesh(os.path.join(
            root_folder, "../data", "spot-degenerate.obj"))
E2, eps2, newL2 = IntrinsicMollification(V2, F2, delta)

print("delta = ", delta)
print("epsilon = ", eps2)
print("Original lengths = ", igl.edge_lengths(V2, F2)[:7])
print("New lengths = ", newL2[:7])


###############################         TESTS           ###############################

####### Test cotanLaplace function
print("______________________________________________________________")
print("Testing cotanLaplace function")

[V,F]= igl.read_triangle_mesh(os.path.join(
            root_folder, "../data", "cow_nonmanifold.obj"))

delta = 0.00000000
E, eps, newL = IntrinsicMollification(V, F, delta)
Laplacian_intrinsic = cotanLaplace(F, newL)

# compute cotan Laplacian using igl library to compare with our intrinsic implementation
Laplacian_igl = igl.cotmatrix(V, F)
# dense matrix <- sparse matrix
Laplacian_igl = np.array(Laplacian_igl.todense())
Laplacian_intrinsic = np.array(Laplacian_intrinsic.todense())

print("Laplacian_intrinsic Dimensions: ",Laplacian_intrinsic.shape)
print("Laplacian_igl Dimensions: ",Laplacian_igl.shape)

print("Laplacian_intrinsic Norm",np.linalg.norm(Laplacian_intrinsic))

print("Laplacian_igl Norm",np.linalg.norm(Laplacian_igl))
print("Norm of Difference = ",np.linalg.norm(Laplacian_intrinsic - Laplacian_igl))

###############################       MASSMATRIX       ###############################

## Test massmatrix function for barycentric type.
print("______________________________________________________________")
print("Testing massmatrix function with barycentric option")

L = igl.edge_lengths(V, F)
mm_igl = igl.massmatrix_intrinsic(L, F, igl.MASSMATRIX_TYPE_BARYCENTRIC)
mm_intrinsic = massmatrix(L, F, MASSMATRIX_TYPE.BARYCENTRIC)

print("IGL massmatrix:\n", mm_igl.todense())
print("\nmassmatrix computed:\n", mm_intrinsic.todense())
print("IGL massmatrix norm: ", sp.sparse.linalg.norm(mm_igl))
print("mass matrix norm: ", sp.sparse.linalg.norm(mm_intrinsic))
print("Norm of difference: ", sp.sparse.linalg.norm(mm_igl - mm_intrinsic))


## Test massmatrix function for circumcentric type.
print("______________________________________________________________")
print("Testing massmatrix function with circumcentric option")

mc_igl = igl.massmatrix_intrinsic(L, F, igl.MASSMATRIX_TYPE_VORONOI)
mc_intrinsic = massmatrix(L, F, MASSMATRIX_TYPE.CIRCUMCENTRIC)

print("IGL massmatrix: \n", mc_igl.todense())
print("\nmassmatrix computed:\n", mc_intrinsic.todense())
print("IGL massmatrix norm: ", sp.sparse.linalg.norm(mc_igl))
print("massmatrix norm: ", sp.sparse.linalg.norm(mc_intrinsic))
print("Norm of difference: ", sp.sparse.linalg.norm(mc_igl - mc_intrinsic))

