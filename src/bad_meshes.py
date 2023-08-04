from Mollification import *
from cotanLaplace import *
from massmatrix import *
import igl

import sys
import os
root_folder = os.getcwd()

def has_zero_area(F, L, epsilon=0):
    areas = doublearea_intrinsic(L)

    for i in range(len(areas)):
        if areas[i] < epsilon:
            return True, i

    return False, -1

def check_mesh_properties(F, L, V):
    non_positive_area, area_index = has_zero_area(F, L, 1e-12)
    if(non_positive_area):
        print("TRIANGLE AREA: Mesh has triangle area at face: ", area_index)
        print("\nApply mollification...")
        m_L, m_eps, m_newL = IntrinsicMollification(V, F, 1e-5)
        print("Mollification epsilon: ", m_eps)
        non_positive_area, area_index = has_zero_area(F, m_newL, 1e-12)
        if(non_positive_area):
            print("There is still non-positive area\n")
        else:
            print("All positive area.\n")
    else:
        print("TRIANGLE AREA: fine.")

def cotmatrix_numerators(F, L):
    Vsize = np.max(F) + 1

    laplacian = sp.sparse.lil_matrix((Vsize, Vsize))

    for face in range(len(F)):
        edges_length = L[face]
        lengths = [edges_length[0], edges_length[1], edges_length[2]]

        for i in range(3):
            j = (i+1) % 3
            k = (i+2) % 3

            wjk = (lengths[j]**2 + lengths[k]**2 - lengths[i]**2)

            fr = F[face]
            laplacian[fr[j], fr[j]] -= wjk
            laplacian[fr[k], fr[k]] -= wjk
            laplacian[fr[j], fr[k]] += wjk
            laplacian[fr[k], fr[j]] += wjk

    return laplacian

def cotmatrix_negative_entry(F, L, nzero=1e-12):
    laplacian = cotmatrix_numerators(F, L)
    Vsize = np.max(F) + 1

    ## Check if there is any non-positive entry:
    for i in range(Vsize):
        for j in range(Vsize):
            if(laplacian[i, j] < (-1)*nzero):
                return True

    return False

def cotmatrix_close_to_zero_entry(F, L, nzero=1e-12):
    laplacian = cotmatrix_numerators(F, L)
    Vsize = np.max(F) + 1

    for i in range(Vsize):
        for j in range(Vsize):
            if(np.abs(laplacian[i, j]) < nzero):
                return True
    return False

def check_cotmatrix_entries(F, L):
    negative_entry = cotmatrix_negative_entry(F, L)
    close_to_zero = cotmatrix_close_to_zero_entry(F, L)

    if(negative_entry):
        print("COTAN-LAPLACE: Mesh would have non-positive cotanLaplacian entry.")
    else:
        print("COTAN-LAPLACE: entries are non-negative.")

    if(close_to_zero):
        print("COTAN-LAPLACE: There is close to zero entry.")
    else:
        print("COTAN-LAPLACE: different from zero entries.")


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "../data/bad_triangle/buser_head.stl"

    V, F = igl.read_triangle_mesh(os.path.join(root_folder, ".", filepath))
    L = igl.edge_lengths(V, F)

    check_mesh_properties(F, L, V)
    check_cotmatrix_entries(F, L)


if __name__ == "__main__":
    main()
