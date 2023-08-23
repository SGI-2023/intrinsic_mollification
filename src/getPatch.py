import numpy as np
import igl
import scipy as sp

from buildGlueMap import *

import meshio
import os
import polyscope as ps



# F: List of faces, a nF x 3 matrix where each entry is a indexed vertex
# G: Gluing map, a nF x 3 x 2 matrix where each entry is a face-side pair
# percentage: Percentage of faces once we reach we stop.
def get_patch(F, G, percentage=0.1):
    # get the number of faces
    nF = np.shape(F)[0]

    # get random face
    f_0 = np.random.randint(0, nF)

    # use BFS to find the patch
    # Standard BFS implementation:
    visited = [f_0]  # List to keep track of visited nodes.
    queue = [f_0]  # Initialize a queue

    while queue and len(visited) < nF * percentage:
        f_i = queue.pop(0)

        for s in range(3):
            fs = (f_i, s)
            fsp = tuple(G[fs])

            if fsp[0] not in visited and fsp[0] != -1:
                visited.append(fsp[0])
                queue.append(fsp[0])

    return F[visited, :]

def main():
    # load the mesh
    mesh = meshio.read("data/spot.obj")
    V = mesh.points
    F = mesh.cells_dict["triangle"]

    # get the gluing map
    G = build_gluing_map(F)

    # get the patch
    patch = get_patch(F, G, 0.7)

    # visualize the patch
    ps.init()
    ps_mesh = ps.register_surface_mesh("patch", V, patch)
    ps.show()

if __name__ == "__main__":
    main()
