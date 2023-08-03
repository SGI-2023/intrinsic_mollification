import igl
import numpy
import scipy as sp
import numpy as np
from enum import Enum

class MASSMATRIX_TYPE(Enum):
    BARYCENTRIC=1
    CIRCUMCENTRIC=2
    CIRCUMCENTRIC_IGL_LIKE=3

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
def massmatrix(uL, F, option=MASSMATRIX_TYPE.BARYCENTRIC):
    L = np.sort(uL, axis=1)
    Vsize = np.max(F)+1
    area = numpy.zeros(Vsize)

    for face in range(len(F)):
        edges_length = L[face]
        e0 = edges_length[0]
        e1 = edges_length[1]
        e2 = edges_length[2]
        
        lengths = [e0, e1, e2]
        for i in range(3):
            fr = F[face]
            if (option  == MASSMATRIX_TYPE.BARYCENTRIC):
                area[fr[i]] += barycentric_area_per_corner(lengths)
            elif (option == MASSMATRIX_TYPE.CIRCUMCENTRIC):
                area[fr[i]] += circumcentric_area_per_corner(lengths, i)
            elif (option == MASSMATRIX_TYPE.CIRCUMCENTRIC_IGL_LIKE):
                area[fr[i]] += circumcentric_area_per_corner_igl_like(lengths, i)

    return sp.sparse.spdiags(area, 0, area.size, area.size)

def barycentric_area_per_corner(l):
    return (1/3)*triangle_area(l)

def triangle_area(l):
    return 0.25*np.sqrt((l[0]+l[1]+l[2])*
                        (l[1]+l[2]-l[0])*
                        (l[0]+l[2]-l[1])*
                        (l[0]+l[1]-l[2]))

def circumcentric_area_per_corner(l, i):
    adj1 = (i+1)%3
    adj2 = (i+2)%3
    opp = i

    to_return = 0

    # Observe that this is not whole angle, we just need the sign to check for 
    #   obtuse triangle.
    # gamma is angle at corner i.
    cot_gamma = l[adj1]**2 + l[adj2]**2 - l[opp]**2
    cot_alpha = l[adj2]**2 + l[opp]**2 - l[adj1]**2
    cot_beta = l[adj1]**2 + l[opp]**2 - l[adj2]**2

    cos_gamma = l[opp] * cot_gamma
    cos_alpha = l[adj1] * cot_alpha
    cos_beta = l[adj2] * cot_beta
    sm = np.sum([cos_gamma, cos_alpha, cos_beta])

    cos_gamma = cos_gamma/sm
    cos_alpha = cos_alpha/sm
    cos_beta = cos_beta/sm

    zero = 1e-12
    # If triangle not obtuse, it is Voronoi safe.
    if ((cot_gamma > zero) and (cot_alpha > zero) and (cot_beta > zero)):
        to_return = ( (l[adj1]**2) * cot_alpha +
                      (l[adj2]**2) * cot_beta )/(32*triangle_area(l))
    elif (cot_gamma < zero): # Angle at i is obtuse.
        to_return = triangle_area(l)/2
    else:
        to_return = triangle_area(l)/4

    return to_return

def circumcentric_area_per_corner_igl_like(l, i):
    adj1 = (i+1)%3
    adj2 = (i+2)%3
    opp = i

    to_return = 0

    # Observe that this is not whole angle, we just need the sign to check for 
    #   obtuse triangle.
    # gamma is angle at corner i.
    cot_gamma = l[adj1]**2 + l[adj2]**2 - l[opp]**2
    cot_alpha = l[adj2]**2 + l[opp]**2 - l[adj1]**2
    cot_beta = l[adj1]**2 + l[opp]**2 - l[adj2]**2

    #### Try normalization. #####
    cos_gamma = cot_gamma/(2*l[adj1]*l[adj2])
    cos_alpha = cot_alpha/(2*l[adj2]*l[opp])
    cos_beta = cot_beta/(2*l[adj1]*l[opp])

    normalized = np.zeros((3))
    normalized[opp] = cos_gamma * l[opp]
    normalized[adj1] = cos_alpha * l[adj1]
    normalized[adj2] = cos_beta * l[adj2]

    normalized = normalized/np.sum(normalized)
    normalized = normalized * triangle_area(l)

    normalized[opp] = (normalized[adj1] + normalized[adj2])/2

    if(cos_gamma < 0):
        return triangle_area(l)/2
    elif ((cos_alpha < 0) or (cos_beta < 0)):
        return triangle_area(l)/4
    else:
        return normalized[opp]

    print("ERROR: incorrect angle at circumcenter_area_per_corner.")
    return -1
