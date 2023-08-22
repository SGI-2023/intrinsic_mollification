import os
import igl
import polyscope as ps
from ex_lscm import *
root_folder = os.getcwd()

if __name__ == "__main__":
    V, F = igl.read_triangle_mesh(os.path.join(root_folder, "../data/", "BunnyHead.obj"))
    done, V_uv = lscm(V, F)

    ps.init()
    ps.register_surface_mesh("mesh", V_uv, F)
    ps.show()
