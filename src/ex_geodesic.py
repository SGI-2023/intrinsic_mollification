import igl

import numpy as np

# from scipy.pysparse import spmatrix

from scipy.sparse import csr_matrix

from Mollification import *
from massmatrix import massmatrix
from massmatrix import doublearea_intrinsic
from cotanLaplace import cotanLaplace

def average_length(V, F):
    newL = IntrinsicMollificationConstant(V, F)[-1]
    avgL = np.mean(newL)
    return avgL


def heat_geodesics_precompute(V, F):
    h = average_length(V, F)
    t = h*h
    return heat_geodesics_precompute2(V, F, t)

def heat_geodesics_precompute2(V, F, t=None):
    # data = igl.HeatGeodesicsData()
    newL = IntrinsicMollificationConstant(V,F)[-1]

    L = cotanLaplace(F, newL)
    M = massmatrix(newL, F)
    dblA = doublearea_intrinsic(newL)
    grad = igl.grad(V, F)

    ng = grad.shape[0] // F.shape[0]
    assert ng == 2 or ng == 3
    div = -0.25 * grad.T @ dblA.repeat(ng)
    
    Q = M - t * L
    
    o = igl.boundary_facets(F)
    b = np.unique(o,axis=0)

    data = {"grad": grad, "div": div, "ng": ng, "Q": Q, "o":o, "b":b, "M": M}
    
    return data


def heat_geodesics_solve(data, gamma):
    n = data["grad"].shape[1]
    u0 = np.zeros((n, 1))

    for g in range(len(gamma)):
        u0[gamma[g]] = 1
     
    Aeq = csr_matrix(np.array([]))   
    
    #neumann
    u = igl.min_quad_with_fixed(data["Q"], u0, np.zeros((data["Q"].shape[0],),dtype=np.int32), np.zeros_like(u0), Aeq, np.zeros(Aeq.shape), True)

    #dirichlet
    if len(data["b"]) > 0:
        uD = np.zeros(len(data["b"]), 1)
        uD = igl.min_quad_with_fixed(data["Q"], u0, data["b"], np.zeros_like(u0), Aeq, np.zeros(Aeq.shape), True)
        u += uD
        u *= 0.5

    grad_u = data["grad"] * u
    m = data["grad"].shape[0] // data["ng"]
    
    for i in range(m):
        norm = 0
        ma = 0
        
        for d in range(data["ng"]):
            ma = np.max(ma, abs(grad_u[d * m + i]))
        
        for d in range(data['ng']):
            gui = grad_u[d * m + i] / ma
            norm += gui * gui
            
        norm = ma * np.sqrt(norm)

        if ma == 0 or norm == 0 or np.isnan(norm):
            grad_u[d * m + i] = 0
        else:
            grad_u[d * m + i] /= norm

    div_X = -data["div"] * grad_u
    Beq = np.array([0])

    
    #poisson
    D = igl.min_quad_with_fixed(data["Q"], -div_X, np.zeros((data["Q"].shape[0],)),None, csr_matrix(np.diag(data["M"])), Beq, True)
    
    Dgamma = D[gamma, :]
    D -= Dgamma.mean()
    if D.mean() < 0:
        D = -D

    return D


# Example usage:
if __name__ == "__main__":
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.double)
    F = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)


    # Precompute the heat geodesics data
    data = heat_geodesics_precompute(V, F)


    # Solve for heat geodesic distances from a set of seed vertices (e.g., gamma)
    gamma = np.array([0], dtype=np.int32)
    D = heat_geodesics_solve(data, gamma)

    print(D)


