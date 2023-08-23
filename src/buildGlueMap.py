'''
Acknowledgement: This code is a modified copy from the code provided by https://github.com/nmwsharp/intrinsic-triangulations-tutorial/blob/master/tutorial_completed.py#L212
'''

import numpy as np

def next_side(fs):
    """
    For a given side s of a triangle, returns the next side t. (This method serves mainly to make code more readable.)

    :param fs: A face side (f,s)
    :returns: The next face side in the same triangle (f, sn)
    """
    return (fs[0], (fs[1]+1)%3)


def other(G, fs):
    """
    For a given face-side fs, returns the neighboring face-side in some other triangle.

    :param G: |F|x3x2 gluing map G,
    :param fs: a face-side (f,s)
    :returns: The neighboring face-side (f_opp,s_opp)
    """
    return tuple(G[fs])

def n_faces(F):
    """
    Return the number of faces in the triangulation.

    :param F: |F|x3 array of face-vertex indices
    :returns: |F|
    """
    return F.shape[0]

def n_verts(F):
    """
    Return the number of vertices in the triangulation.

    Note that for simplicity this function recovers the number of vertices from
    the face listing only. As a consequence it is _not_ constant-time, and
    should not be called in a tight loop.

    :param F: |F|x3 array of face-vertex indices
    :returns: |F|
    """
    return np.amax(F)+1

def sort_rows(A):
    """
    Sorts rows lexicographically, i.e., comparing the first column first, then
    using subsequent columns to break ties.

    :param A: A 2D array
    :returns: A sorted array with the same dimensions as A
    """
    return A[np.lexsort(np.rot90(A))]


def glue_together(G, fs1, fs2):
    """
    Glues together the two specified face sides.  Using this routine (rather
    than manipulating G directly) just helps to ensure that a basic invariant
    of G is always preserved: if a is glued to b, then b is glued to a.

    The gluing map G is updated in-place.

    :param G: |F|x3x2 gluing map
    :param fs1: a face-side (f1,s1)
    :param fs2: another face-side (f2,s2)
    """
    G[fs1] = fs2
    G[fs2] = fs1

def glue_boundary(G, fs):
    """
    Glues the specified face-side to the boundary.
    """
    G[fs] = (-1,-1)


def build_gluing_map(F):
    """
    Builds the gluing map for a triangle mesh.

    :param F: |F|x3 vertex-face adjacency list F describing a manifold, oriented triangle mesh.
    :returns: |F|x3x2 gluing map G, which for each side of each face stores the
    face-side it is glued to.  In particular, G[f,s] is a pair (f',s') such
    that (f,s) and (f',s') are glued together.
    """
    
    # In order to construct this array, for each side of a triangle, we need to
    # find the neighboring side in some other triangle. There are many ways that
    # this lookup could be accomplished. Here, we use an array-based strategy
    # which constructs an `Sx4` array (where `S` is the number of face-sides),
    # where each row holds the vertex indices of a face-side, as well as the face
    # it comes from and which side it is. We then sort the rows of this array
    # lexicographically, which puts adjacent face-sides next to each other in the
    # sorted array. Finally, we walk down the array and populate the gluing map
    # with adjacent face-side entries.


    # Build a temporary list S of all face-sides, given by tuples (i,j,f,s),
    # where (i,j) are the vertex indices of side s of face f in sorted order
    # (i<j).

    n_sides = 3*n_faces(F)
    S = np.empty([n_sides,4], dtype=np.int64)

    for f in range(n_faces(F)):    # iterate over triangles
        for s in range(3):         # iterate over the three sides

            # get the two endpoints (i,j) of this side, in sorted order
            i = F[f,s]
            j = F[next_side((f,s))]
            S[f*3+s] = (min(i,j),max(i,j),f,(s+2)%3)

    # Sort the list row-wise (so i-j pairs are adjacent)
    S = sort_rows(S)

    # save S to a text file for debugging
    # np.savetxt("S.txt", S, fmt="%d")

    # Build the |F|x3 gluing map G, by linking together pairs of sides with the same vertex indices.
    G = np.empty([n_faces(F),3,2], dtype=np.int64)
    p = 0
    while p < n_sides:
        # handle the case where the side is on the boundary
        if p == n_sides-1 or S[p+0,0] != S[p+1,0] or S[p+0,1] != S[p+1,1]:
            fs = tuple(S[p+0,2:4])
            glue_boundary(G, fs)
            p += 1

        else:
            fs0 = tuple(S[p+0,2:4])
            fs1 = tuple(S[p+1,2:4])
            glue_together(G, fs0, fs1)
            p += 2

    # A sanity-check test
    validate_gluing_map(G)

    return G


def validate_gluing_map(G):
    """
    Performs sanity checks on the connectivity of the gluing map. Throws an
    exception if anything is wrong.

    :param G: |F|x3x2 gluing map G
    """

    for f in range(n_faces(G)):
        for s in range(3):

            fs = (f,s)

            if other(G, fs)[0] == -1:
                # this is a boundary face-side, so skip it
                continue

            fs_other = other(G, fs)

            if fs == fs_other:
                raise ValueError("gluing map points face-side to itself {}".format(fs))

            if fs != other(G, fs_other):
                raise ValueError("gluing map is not involution (applying it twice does not return the original face-side) {} -- {} -- {}".format(fs, fs_other, other(G, fs_other)))
