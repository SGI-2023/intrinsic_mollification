import os
import igl
import sys
import polyscope as ps
from ex_lscm import *
root_folder = os.getcwd()

if __name__ == "__main__":
    filename = ""
    if(len(sys.argv) != 2):
        filename = "../data/BunnyHead.obj"
    else:
        filename = sys.argv[1]

    V, F = igl.read_triangle_mesh(os.path.join(root_folder, ".", filename))
    done, V_uv = lscm(V, F)

    ps.init()
    ps.register_surface_mesh("mesh", V_uv, F)
    ps.show()
