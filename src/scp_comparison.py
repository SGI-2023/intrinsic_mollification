import polyscope as ps
import igl
from ex_lscm import *

import os
import sys
root_folder = os.getcwd() 

def main():
    filepath = "../data/spot.obj" if len(sys.argv) < 2 else sys.argv[1]
    V, F = igl.read_triangle_mesh(os.path.join(root_folder, ".", filepath))

    eigvec, eigval, V_uv = scp(V, F, "to_abs", "to_zero")

    ps.init()
    ps.register_surface_mesh("mesh", V_uv, F)
    #ps.get_surface_mesh("mesh").add_parameterization_quantity("LSCM", V_uv)
    ps.show()

if __name__ == "__main__":
    main()
