import os
import numpy as np
import igl
import polyscope as ps
ps.init()


# Point Id of the vertex source
Id = 38934  # ear of the bunny :3
#Id = 0

# Load mesh
root_folder = os.getcwd()
path = "Thingi10k/raw_meshes/"
V, F = igl.read_triangle_mesh( os.path.join( root_folder, "../data", "bunny.obj"))

## Select a vertex from which the distances should be calculated
vs = np.array([Id])
vt = np.arange( len(V) )


# Compute distace
d = igl.exact_geodesic(V, F, vs = vs, vt = vt) 
print(d)

ps_mesh = ps.register_surface_mesh("my mesh", V, F)
ps_mesh.add_distance_quantity("distances", d, enabled=True, stripe_size=0.009)
ps.show() 
