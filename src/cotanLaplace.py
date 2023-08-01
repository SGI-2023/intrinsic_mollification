'''
    Phase II:
    Build the cotan-Laplace matrix from the intrinsic lengths.
    (A description of how to do this can be found in the same paper by Sharp & Crane).
'''
import scipy as sp
import numpy as np
import igl

'''
    Parameters:
        F: List of faces, a N x 3 matrix where each entry is a indexed vertex
        L: Lengths of edges in the order of edges [1, 2], [2, 0], [0, 1].
        Vsize: number of vertices.
    Returns:
        Laplacian: Laplacian in format of a lil_matrix from scipy sparse matrix.
'''
def cotanLaplace(F, L, Vsize):

    LaplacianMollified = sp.sparse.lil_matrix((Vsize, Vsize))

    for face in range(len(F)):
        edges_length = L[face]
        e0 = edges_length[0]
        e1 = edges_length[1]
        e2 = edges_length[2]

        s = (e0 + e1 + e2)/2
        area = np.sqrt(s*(s-e1)*(s-e2)*(s-e0))

        lengths = [e0, e1, e2]
        for i in range(3): # Consider face ijk
            # Get w per corner:
            wjk = (lengths[(i+1)%3] ** 2 + lengths[(i+2)%3] ** 2 - lengths[i] ** 2) / (8*area)

            # Update the cotan-Laplace matrix.
            fr = F[face]
            LaplacianMollified[fr[(i+1)%3], fr[(i+1)%3]] -= wjk #L_jj
            LaplacianMollified[fr[(i+2)%3], fr[(i+2)%3]] -= wjk #L_kk
            LaplacianMollified[fr[(i+1)%3], fr[(i+2)%3]] += wjk #L_jk
            LaplacianMollified[fr[(i+2)%3], fr[(i+1)%3]] += wjk #L_kj

    return LaplacianMollified
