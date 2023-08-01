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
            root_folder, "data", "spot-degenerate.obj"))
E2, newL2, eps2 = IntrinsicMollification(V2, F2, delta)

print("delta = ", delta)
print("epsilon = ", eps2)
print("Original lengths = ", igl.edge_lengths(V2, E2)[:7])
print("New lengths = ", newL2[:7])


###############################         TESTS           ###############################

####### Test cotanLaplace function
annulus = trimesh.creation.annulus(r_min = 1, r_max = 3, height = 5)
V = np.array(annulus.vertices)
F = np.array(annulus.faces)

delta = 0.2
E, newL, eps = cotanLaplace(V, F, delta)

print("delta = ", delta)
print("epsilon = ", eps)
print("Original lengths = ", igl.edge_lengths(V, E)[:7])
print("New lengths = ", newL[:7])