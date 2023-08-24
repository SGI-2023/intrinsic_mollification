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
    done, V_uv = lscm(V, F,
                      mollified=False,
                      neg_hack=NEG_HACK.NONE,
                      nan_hack=NAN_HACK.NONE,
                      close_zero_hack=CLOSE_TO_ZERO_HACK.NONE)

    done, V_uv_neg = lscm(V, F, 
                          mollified=False,
                          neg_hack=NEG_HACK.TO_ABS,
                          nan_hack=NAN_HACK.NONE,
                          close_zero_hack=CLOSE_TO_ZERO_HACK.NONE)

    done, V_uv_zero = lscm(V, F,
                           mollified=False,
                           neg_hack=NEG_HACK.NONE,
                           nan_hack=NAN_HACK.NONE,
                           close_zero_hack=CLOSE_TO_ZERO_HACK.TO_ZERO)

    done, V_uv_mol = lscm(V, F,
                           mollified=True,
                           neg_hack=NEG_HACK.NONE,
                           nan_hack=NAN_HACK.NONE,
                           close_zero_hack=CLOSE_TO_ZERO_HACK.NONE)

    print("NUMBER OF FLIPPED TRUANGLES:")
    print("Original: ", igl.flipped_triangles(V_uv, F).size)
    print("Mollified: ", igl.flipped_triangles(V_uv_mol, F).size)
    print("Negative hack: ", igl.flipped_triangles(V_uv_neg, F).size)
    print("Zero hack: ", igl.flipped_triangles(V_uv_zero, F).size)

    ps.init()
    ps.register_surface_mesh("mesh", V_uv_mol, F, edge_width=1) 
    ps.show()
