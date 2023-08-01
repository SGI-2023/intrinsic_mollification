import igl
import numpy
import scipy as sp
import numpy as np

'''
Python bindings for libigl does not include intrinsic doublearea. 

This function replicates intrinsic doublearea from C++.

Inputs:
    ul: edge_lengths of size #F x 3 assuming triangles.
Outputs:
    dblA: #F list of triangles double area.
''' 
def doublearea_intrinsic(ul):
    dblA = numpy.zeros(len(l))
    for i in range(len(l)):
        edges = l[i]
        e0 = edges[0]
        e1 = edges[1]
        e2 = edges[2]

        a = 2*0.25*np.sqrt((e0+e1+e2) *
                           (e1+e2-e0) * 
                           (e0+e2-e1) *
                           (e0+e1-e2) )

        dblA[i] = a

    return dblA

'''
Based on implementation of c++ libigl massmatrix_intrinsic.
'''
def massmatrix(uL, F):
    L = np.sort(uL, axis=1)
    Vsize = np.max(F)+1
    area = numpy.zeros(Vsize)

    for face in range(len(F)):
        edges_length = L[face]
        e0 = edges_length[0]
        e1 = edges_length[1]
        e2 = edges_length[2]
        
        a = 2*0.25*np.sqrt((e0+e1+e2) *
                           (e1+e2-e0) * 
                           (e0+e2-e1) *
                           (e0+e1-e2) )

        for i in range(3):
            fr = F[face]
            area[fr[i]] += a
    area = area/6
    return sp.sparse.spdiags(area, 0, area.size, area.size)
