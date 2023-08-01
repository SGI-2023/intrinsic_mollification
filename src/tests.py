import trimesh
import os
import numpy as np
import igl
from Mollification import *
from cotanLaplace import *



root_folder = os.getcwd()

###############################         TESTS           ###############################           

####### Test IntrinsicMollification function

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
delta = 1e-4
V2, F2 = igl.read_triangle_mesh(os.path.join(
            root_folder, "../data", "spot-degenerate.obj"))
E2, newL2, eps2 = IntrinsicMollification(V2, F2, delta)

print("delta = ", delta)
print("epsilon = ", eps2)
print("Original lengths = ", igl.edge_lengths(V2, E2)[:7])
print("New lengths = ", newL2[:7])


###############################         TESTS           ###############################

####### Test cotanLaplace function
[V,F]= igl.read_triangle_mesh(os.path.join(
            root_folder, "../data", "cow_nonmanifold.obj"))

delta = 0.00000000
E, newL, eps = IntrinsicMollification(V, F, delta)
Laplacian_intrinsic = cotanLaplace(F, E, newL, V.shape[0])

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
