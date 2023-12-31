'''
    Phase I:
    Implement the intrinsic mollification scheme by
    (i) reading off the original edge lengths and
    (ii) implementing the formula defined by Sharp & Crane to get new lengths that describe higher-quality triangles.

    Phase ??:
    Adding Multiple Mollification Schemes as described Below

'''

import igl
import numpy as np
import scipy as sp
import cvxopt as cvx
from buildGlueMap import *

from enum import Enum


'''
Generalized IntrinsicMollification:
    - Constant Epsilon (0)
    - Local schemes (1)
        - One-By-One Step
        - One-By-One Interpolated
        - Local Least-Mollification (Manhattan (L1) -> Lin  Prog)
        - Local Least-Mollification (Euclidean (L2) -> Quad Prog)

    - (ARAP-like) Sequential Global Schemes (Get local, Pool, ...) (2)
        - One-By-One Step
        - One-By-One Interpolated
        - Local Least-Mollification (Manhattan (L1) -> Lin  Prog)
        - Local Least-Mollification (Euclidean (L2) -> Quad Prog)

    - For these, TODO: try to normalize edge lengthes (to preserve total surface area) after each loop before pooling

    - Global Optimization Schemes (3, 4)
        - Global Least-Mollification (Manhattan (L1) -> Lin  Prog) (3)
        - Global Least-Mollification (Euclidean (L2) -> Quad Prog) (4)

    - Pooling Options for Sequential Global Schemes
        - Mean
        - Max

    - Options for delta factor
        - Mean Edge Length
        - Min Edge Length

    - Optional: Scale edge lengths down to perserve total area

Methods to Compare Performance:
- Avg Estimated Time (on ~1k meshes)
- Avg Epsilon per mesh normalized by Mean Edge Length -> Average this statistic over all meshes
- Avg Iterations to Convergence per mesh (For (ARAP-like) Sequential Global Schemes)

- Visualize the difference on a bunch of applications and meshes:
    - Conformal Parameterization (LSCM/SCP)
    - Laplacian Smoothing (Mean Curvature Flow)
    - Geodesic Distance (Heat Method)
    - Simple Poisson Solver (interpolate values)
    - Manifold Harmonics
- For each application, try to find a metric of accuracy or a ground truth to compare to in a quantitative manner
'''

class MOLLIFICATION_SCHEME(Enum):
    CONSTANT_EPSILON = 0
    LOCAL_SCHEMES = 1
    SEQUENTIAL_GLOBAL = 2
    GLOBAL_OPTIMIZATION_MANHATTAN = 3
    GLOBAL_OPTIMIZATION_EUCLIDEAN = 4

class MOLLIFICATION_LOCAL_SCHEME(Enum):
    ONE_BY_ONE_STEP = 0
    ONE_BY_ONE_INTERPOLATED = 1
    LOCAL_LEAST_MOLLIFICATION_MANHATTAN = 2
    LOCAL_LEAST_MOLLIFICATION_EUCLIDEAN = 3
    CONSTANT_EPSILON = 4

class MOLLIFICATION_DELTA_FACTOR(Enum):
    MEAN_EDGE_LENGTH = 0
    MIN_EDGE_LENGTH = 1

class MOLLIFICATION_POOLING(Enum):
    MEAN = 0
    MAX = 1

def CheckInequalityLocal(L, delta = 1e-4, threshold = 1e-4):
    return max(delta + L[0] - L[1] - L[2], delta - L[0] + L[1] - L[2], delta - L[0] - L[1] + L[2]) < threshold * delta

def CheckInequalityGlobal(L, delta = 1e-4, threshold = 1e-4):
    print (np.max( [delta + L[:,0] - L[:,1] - L[:,2], delta - L[:,0] + L[:,1] - L[:,2], delta - L[:,0] - L[:,1] + L[:,2] ]  ), threshold * delta)
    return np.max( [delta + L[:,0] - L[:,1] - L[:,2], delta - L[:,0] + L[:,1] - L[:,2], delta - L[:,0] - L[:,1] + L[:,2] ]  ) < threshold * delta

'''
    Parameters:
        FL: List of face edge lengths, a #F x 3 matrix where each entry is a edge length
    Optional Parameters:
        delta: such that l_ij + l_jk > l_ki + delta (for all edges)
        scheme: which mollification scheme to use. This describes how the algorithm will progress.
        local_scheme: which local mollification scheme to use. This describes how to mollify each triangle.
        delta_factor_type: determines the multiplier for delta. This describes how to compute delta (mean or min edge length).
        pooling: for sequential global schemes, this describes how to pool the local mollifications so that the halfedge lengths match up (mean or max).
        total_area_preservation: whether or not to scale the edge lengths down to preserve total area.
    Returns:
        newFL: new edges lengths in the same shape as FL

'''
def IntrinsicMollificationFL(FL, G = None, delta = 1e-4,
                             scheme = MOLLIFICATION_SCHEME.CONSTANT_EPSILON,
                             local_scheme = MOLLIFICATION_LOCAL_SCHEME.ONE_BY_ONE_STEP,
                             delta_factor_type = MOLLIFICATION_DELTA_FACTOR.MEAN_EDGE_LENGTH,
                             pooling = MOLLIFICATION_POOLING.MEAN,
                             total_area_preservation = False):
    delta = delta * np.mean(FL) if delta_factor_type == MOLLIFICATION_DELTA_FACTOR.MEAN_EDGE_LENGTH else delta * np.min(FL)

    if scheme == MOLLIFICATION_SCHEME.CONSTANT_EPSILON:
        newFL, eps = IntrinsicMollification_Constant(FL, delta)
        nMoll = np.shape(FL)[0] if eps > 1e-6 * delta else 0
        nIter = 1
    elif scheme == MOLLIFICATION_SCHEME.LOCAL_SCHEMES:
        newFL = np.copy(FL)
        newFL, nMoll = IntrinsicMollification_Local(newFL, delta, local_scheme)
        nIter = 1
    elif scheme == MOLLIFICATION_SCHEME.SEQUENTIAL_GLOBAL:
        newFL = np.copy(FL)
        newFL, nMoll, nIter = IntrinsicMollification_Sequential_Global(newFL, G, delta, local_scheme, pooling)
    elif scheme == MOLLIFICATION_SCHEME.GLOBAL_OPTIMIZATION_MANHATTAN:
        newFL = np.copy(FL)
        #newFL, nMoll = IntrinsicMollification_Global_Optimization_Manhattan(newFL, G, delta)
        newFL, nMoll = IntrinsicMollification_Global_Optimization_Manhattan(newFL, G, delta)
        nIter = 1
    elif scheme == MOLLIFICATION_SCHEME.GLOBAL_OPTIMIZATION_EUCLIDEAN:
        newFL = np.copy(FL)
        newFL, nMoll = IntrinsicMollification_Global_Optimization_Euclidean(newFL, G, delta)
        nIter = 1

    if total_area_preservation:
        pass # TODO: Scale edge lengths down to perserve total area

    return newFL, nMoll, nIter


'''
    Parameters:
        V: List of coordinates vertices, #V x 3 matrix
        F: List of faces, a #F x 3 matrix where each entry is a indexed vertex
    Optional Parameters:
        delta: such that l_ij + l_jk > l_ki + delta (for all edges)
        scheme: which mollification scheme to use. This describes how the algorithm will progress.
        local_scheme: which local mollification scheme to use. This describes how to mollify each triangle.
        delta_factor_type: determines the multiplier for delta. This describes how to compute delta (mean or min edge length).
        pooling: for sequential global schemes, this describes how to pool the local mollifications so that the halfedge lengths match up (mean or max).
        total_area_preservation: whether or not to scale the edge lengths down to preserve total area.
    Returns:
        newFL: new edges lengths in the same shape as FL
'''
def IntrinsicMollification(V, F, delta = 1e-4,
                           scheme = MOLLIFICATION_SCHEME.CONSTANT_EPSILON,
                           local_scheme = MOLLIFICATION_LOCAL_SCHEME.ONE_BY_ONE_STEP,
                           delta_factor_type = MOLLIFICATION_DELTA_FACTOR.MEAN_EDGE_LENGTH,
                           pooling = MOLLIFICATION_POOLING.MEAN,
                           total_area_preservation = False):
    FL = igl.edge_lengths(V, F)         # columns correspond to edges lengths [1,2],[2,0],[0,1]
    #get Glue map
    if scheme == MOLLIFICATION_SCHEME.SEQUENTIAL_GLOBAL or                          \
        scheme == MOLLIFICATION_SCHEME.GLOBAL_OPTIMIZATION_EUCLIDEAN or             \
        scheme == MOLLIFICATION_SCHEME.GLOBAL_OPTIMIZATION_MANHATTAN:
        G = build_gluing_map(F)
    else:
        G = None

    return IntrinsicMollificationFL (FL, G, delta, scheme, local_scheme, delta_factor_type, pooling, total_area_preservation)


'''
    Parameters:
        FL: List of face edge lengths, a #F x 3 matrix where each entry is a edge length

    Optional Parameters:
        delta: such that l_ij + l_jk > l_ki + delta (for all edges), Note delta is scaled by the mean (or min) edge length

    Returns:
        newL: new edges lengths in the same shape as FL (a #F x 3 matrix where each entry is a edge length)
'''

def IntrinsicMollification_Constant(FL, delta = 1e-4):
    eps = 0.0

    # replaced the loop above with np operations (vectorized, so much faster)
    eps = max(np.max( [delta + FL[:,0] - FL[:,1] - FL[:,2], delta - FL[:,0] + FL[:,1] - FL[:,2], delta - FL[:,0] - FL[:,1] + FL[:,2] ]  ), 0)

    newL = eps + FL
    #assert CheckInequalityGlobal(newL, delta)
    return newL, eps


def IntrinsicMollification_Local(FL, delta = 1e-4,
                                 local_scheme = MOLLIFICATION_LOCAL_SCHEME.ONE_BY_ONE_STEP):
    if local_scheme == MOLLIFICATION_LOCAL_SCHEME.ONE_BY_ONE_STEP:
        return IntrinsicMollification_Local_OneByOneStep(FL, delta)
    elif local_scheme == MOLLIFICATION_LOCAL_SCHEME.ONE_BY_ONE_INTERPOLATED:
        return IntrinsicMollification_Local_OneByOneInterpolated(FL, delta)
    elif local_scheme == MOLLIFICATION_LOCAL_SCHEME.LOCAL_LEAST_MOLLIFICATION_MANHATTAN:
        return IntrinsicMollification_Local_LocalLeastManhattan(FL, delta)
    elif local_scheme == MOLLIFICATION_LOCAL_SCHEME.LOCAL_LEAST_MOLLIFICATION_EUCLIDEAN:
        return IntrinsicMollification_Local_LocalLeastEuclidean(FL, delta)
    elif local_scheme == MOLLIFICATION_LOCAL_SCHEME.CONSTANT_EPSILON:
        return IntrinsicMollification_Local_Constant(FL, delta)


def IntrinsicMollification_Sequential_Global(FL, G, delta = 1e-4,
                                                local_scheme = MOLLIFICATION_LOCAL_SCHEME.ONE_BY_ONE_STEP,
                                                pooling = MOLLIFICATION_POOLING.MEAN):
    nMoll = 0
    currMoll = 1 # dummy value to enter the loop
    i = 0

    while currMoll > 0:
        currMoll = 0

        newFL, currMoll = IntrinsicMollification_Local(FL, delta, local_scheme)

        # use the gluing map to pool the local mollifications, G(f,s) = (f',s'), where f' is the face glued to f along side s
        # we need to set FL(f,s) = FL(f',s') = mean(FL(f,s), FL(f',s')) (or max(FL(f,s), FL(f',s')) if pooling == MOLLIFICATION_POOLING.MAX)
        for f in range(len(FL)):
            for s in range(3):
                fs = (f,s)
                fsp = tuple(G[fs])
                #print(type(fs), type(fsp))
                #print(fs, fsp)
                if fsp[0] == -1 or fsp[1] == -1:
                    continue

                if pooling == MOLLIFICATION_POOLING.MEAN:
                    newVal = 0.5 * (newFL[fs] + newFL[fsp])
                    #print(newVal, newFL[fs], newFL[fsp])
                    newFL[fs] = newVal
                    newFL[fsp] = newVal

                elif pooling == MOLLIFICATION_POOLING.MAX:
                    newVal = max(newFL[fs], newFL[fsp])
                    newFL[fs] = newVal
                    newFL[fsp] = newVal

        i += 1
        nMoll += currMoll
        if i > 100:
            print(i, currMoll, nMoll)

    return newFL, nMoll, i


def IntrinsicMollification_Local_Constant(L, delta = 1e-4):

    nMoll = 0

    for i in range(len(L)):
        if CheckInequalityLocal(L[i], delta):
            continue

        # apply constant mollification
        eps = max(np.max( [delta + L[i][0] - L[i][1] - L[i][2], delta - L[i][0] + L[i][1] - L[i][2], delta - L[i][0] - L[i][1] + L[i][2] ]  ), 0)

        L[i] += eps

        nMoll += 1

    #assert CheckInequalityGlobal(L, delta)

    return L , nMoll

def IntrinsicMollification_Local_OneByOneStep(L, delta = 1e-4):

    nMoll = 0

    for i in range(len(L)):
        if CheckInequalityLocal(L[i], delta):
            continue

        # reorder a, b, c so that c <= b <= a
        L_index = np.argsort(L[i])
        a = L[i][L_index[2]]
        b = L[i][L_index[1]]
        c = L[i][L_index[0]]

        # (step) mollify
        c = max(c, delta + a - b, delta + b - a)
        b = max(b, c)
        a = max(a, b)

        # reorder back to original order
        L[i][L_index[0]] = c
        L[i][L_index[1]] = b
        L[i][L_index[2]] = a

        nMoll += 1

    #assert CheckInequalityGlobal(L, delta)

    return L , nMoll


def IntrinsicMollification_Local_OneByOneInterpolated(L, delta = 1e-4):

    nMoll = 0

    for i in range(len(L)):
        if CheckInequalityLocal(L[i], delta):
            continue

        # reorder a, b, c so that c <= b <= a
        L_index = np.argsort(L[i])
        a = L[i][L_index[2]]
        b = L[i][L_index[1]]
        c = L[i][L_index[0]]

        # (interpolated) mollify
        c_prev = c
        c = max(c, delta + a - b, delta + b - a)
        if a - c_prev > 1e-6 * delta:
            b = max(c, b + (b - c_prev) / (a - c_prev) * delta)
        else:
            b = max(b, c)
        a = max(a, b)

        # reorder back to original order
        L[i][L_index[0]] = c
        L[i][L_index[1]] = b
        L[i][L_index[2]] = a

        nMoll += 1

    #assert CheckInequalityGlobal(L, delta)

    return L , nMoll

def IntrinsicMollification_Local_LocalLeastManhattan(L, delta = 1e-4):
    # we want to minimize the Manhattan distance between the original and new edge lengths
    # we can do this by minimizing the sum of the absolute values of the differences, which is a linear program

    # min C'x
    C = np.ones(3) # C = [1, 1, 1]

    # s.t. Ax <= b
    # a + b >= c + delta, b + c >= a + delta, c + a >= b + delta, a>=a_0, b>=b_0, c>=c_0
    # rewrite 1<->3 as: -a - b + c <= -delta, -b - c + a <= -delta, -c - a + b <= -delta
    A = np.array([[-1, -1, 1],
                  [1, -1, -1],
                  [-1, 1, -1],
                  [-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]])

    b = np.array([- delta,   - delta,    - delta,    0,     0,    0])

    nMoll = 0

    for i in range(len(L)):
        # see if the triangle is already good
        if CheckInequalityLocal(L[i], delta):
            continue

        #print(L[i])
        b[3] = -L[i][0]
        b[4] = -L[i][1]
        b[5] = -L[i][2]
        # solve
        L[i] = sp.optimize.linprog(C, A, b).x

        nMoll += 1

    #assert CheckInequalityGlobal(L, delta)

    return L , nMoll

def BuildFaceAndEdgeMaps(FL, G, delta = 1e-4):
    # we need to build E2FL & FL2E, initially set to -1
    E2FL = np.full((len(FL) * 3, 2), -1, dtype=int)
    FL2E = np.full((len(FL), 3), -1, dtype=int)

    iE = 0

    for i in range(len(FL)):
        # Optimization: we can't skip faces that are already good because their side lengths may change from the mollification neighboring faces
        # but we can skip faces that are already good and also have already good neighbors
        # TODO: Prove or disprove that this optimization is correct, i.e. that there won't be mismatched edges
        # if CheckInequalityLocal(FL[i], delta):
        #     for j in range(3):
        #         fs = (i,j)
        #         fsp = tuple(G[fs])
        #         skipFlag = True

        #         if not ((fsp[0] == -1 or fsp[1] == -1) or CheckInequalityLocal(FL[fsp[0]], delta)):
        #             skipFlag = False

        #     if skipFlag:
        #         continue

        for j in range(3):
            fs = (i,j)
            fsp = tuple(G[fs])

            if fsp[0] == -1 or fsp[1] == -1: # boundary edge so no need to check if we have already added it
                E2FL[iE,0] = fs[0]
                E2FL[iE,1] = fs[1]
                FL2E[fs] = iE
                iE += 1

            else:
                if FL2E[fs] == -1:
                    E2FL[iE,0] = fs[0]
                    E2FL[iE,1] = fs[1]
                    FL2E[fs] = iE
                    FL2E[fsp] = iE
                    iE += 1

    E2FL = E2FL[:iE]

    return E2FL, FL2E

def IntrinsicMollification_Global_Optimization_Manhattan(FL, G, delta = 1e-4):
    # we want to minimize the Manhattan distance between the original and new edge lengths
    # we can do this by minimizing the sum of the absolute values of the differences, which is a linear program

    # we need to build E2FL & FL2E, initially set to -1
    E2FL, FL2E = BuildFaceAndEdgeMaps(FL, G, delta)


    # min C'x
    C = np.ones(np.shape(E2FL)[0]) # C = [1, 1, 1, ...]

    # s.t. Ax <= b
    # a + b >= c + delta, b + c >= a + delta, c + a >= b + delta, a>=a_0, b>=b_0, c>=c_0
    # rewrite 1<->3 as: -a - b + c <= -delta, -b - c + a <= -delta, -c - a + b <= -delta
    A = sp.sparse.lil_matrix((np.shape(E2FL)[0] * 3, np.shape(E2FL)[0]), dtype=float)
    # .zeros((np.shape(E2FL)[0] * 3, np.shape(E2FL)[0]), dtype=float)
    b = np.zeros(np.shape(E2FL)[0] * 3, dtype=float)

    iA = 0
    nMoll = 0

    # s.t.

    for i in range(len(E2FL)):
        fs = (E2FL[i][0], E2FL[i][1])
        fsp = tuple(G[fs])

        iL1 = i
        iL2 = FL2E[fs[0], (fs[1] + 1) % 3]
        iL3 = FL2E[fs[0], (fs[1] + 2) % 3]

        A[iA, iL1] = 1
        A[iA, iL2] = -1
        A[iA, iL3] = -1

        b[iA] = - delta

        iA += 1
        nMoll += 1

        # print("i: ", i, "fs: ", fs, "fsp: ", fsp, "iL1: ", iL1, "iL2: ", iL2, "iL3: ", iL3)

        if fsp[0] != -1 and fsp[1] != -1:
            iL1 = i
            iL2 = FL2E[fsp[0], (fsp[1] + 1) % 3]
            iL3 = FL2E[fsp[0], (fsp[1] + 2) % 3]

            A[iA, iL1] = 1
            A[iA, iL2] = -1
            A[iA, iL3] = -1

            b[iA] = - delta

            iA += 1
            nMoll += 1

            # print("i: ", i, "fs: ", fs, "fsp: ", fsp, "iL1: ", iL1, "iL2: ", iL2, "iL3: ", iL3)

        A[iA, i] = -1
        b[iA] = - FL[fs[0], fs[1]]

        # print("i: ", i, "fs: ", fs, "fsp: ", fsp, "iL1: ", iL1, "iL2: ", iL2, "iL3: ", iL3, "FL[fs[0], fs[1]]: ", FL[fs[0], fs[1]])

        iA += 1

    # slice A and b to remove extra rows based on iA
    A = A[:iA]
    b = b[:iA]

    #np.savetxt("A.txt", A)

    # solve
    LE = sp.optimize.linprog(C, A, b).x
    #print(np.shape(LE), LE)

    # update FL
    FS = (E2FL[:,0], E2FL[:,1])
    FL[FS] = LE[FL2E[FS]]

    FSP = G[FS][((G[FS][:,0] != -1) & (G[FS][:,1] != -1)), :]
    FSP = (FSP[:, 0], FSP[:, 1])
    FL[FSP] = LE[FL2E[FSP]]

    nMoll = nMoll // 3

    #assert CheckInequalityGlobal(FL, delta)
    return FL, nMoll

def IntrinsicMollification_Global_Optimization_Euclidean(FL, G, delta = 1e-4):
    # we want to minimize the Manhattan distance between the original and new edge lengths
    # we can do this by minimizing the sum of the absolute values of the differences, which is a linear program

    # we need to build E2FL & FL2E, initially set to -1
    E2FL, FL2E = BuildFaceAndEdgeMaps(FL, G, delta)

    cvx.solvers.options['show_progress'] = False

    # min 1/2 * x'Cx + q'x
    C = cvx.spmatrix(1.0, range(np.shape(E2FL)[0]), range(np.shape(E2FL)[0])) # C = [1, 1, 1, ...]

    q = cvx.matrix(np.zeros((np.shape(E2FL)[0],1), dtype=float)) # 3x1 zero vector

    # s.t. Ax <= b
    # a + b >= c + delta, b + c >= a + delta, c + a >= b + delta, a>=a_0, b>=b_0, c>=c_0
    # rewrite 1<->3 as: -a - b + c <= -delta, -b - c + a <= -delta, -c - a + b <= -delta

    # for cvxopt, we need to store 3 arrays: val, row, col and then construct the sparse matrix later
    A_Val = np.zeros(np.shape(E2FL)[0] * 7, dtype=float) # atmost 7 non-zero entries per edge (3 for each face, 1 for the edge itself)
    A_Row = np.zeros(np.shape(E2FL)[0] * 7, dtype=int)
    A_Col = np.zeros(np.shape(E2FL)[0] * 7, dtype=int)
    B_Val = np.zeros(np.shape(E2FL)[0] * 3, dtype=float)

    #A = sp.sparse.lil_matrix((np.shape(E2FL)[0] * 3, np.shape(E2FL)[0]), dtype=float)
    #b = np.zeros(np.shape(E2FL)[0] * 3, dtype=float)

    iA = 0
    iSp = 0
    nMoll = 0

    # s.t.

    for i in range(len(E2FL)):
        fs = (E2FL[i][0], E2FL[i][1])
        fsp = tuple(G[fs])

        iL1 = i
        iL2 = FL2E[fs[0], (fs[1] + 1) % 3]
        iL3 = FL2E[fs[0], (fs[1] + 2) % 3]

        # A[iA, iL1] = 1
        # A[iA, iL2] = -1
        # A[iA, iL3] = -1
        # B[iA] = Lj + Lk - Li - delta
        A_Row[iSp] = iA
        A_Col[iSp] = iL1
        A_Val[iSp] = 1
        iSp += 1

        A_Row[iSp] = iA
        A_Col[iSp] = iL2
        A_Val[iSp] = -1
        iSp += 1

        A_Row[iSp] = iA
        A_Col[iSp] = iL3
        A_Val[iSp] = -1

        Li = FL[fs[0], fs[1]]
        Lj = FL[fs[0], (fs[1] + 1) % 3]
        Lk = FL[fs[0], (fs[1] + 2) % 3]
        B_Val[iA] = Lj + Lk - Li - delta

        iSp += 1

        iA += 1
        nMoll += 1

        # print("i: ", i, "fs: ", fs, "fsp: ", fsp, "iL1: ", iL1, "iL2: ", iL2, "iL3: ", iL3)

        if fsp[0] != -1 and fsp[1] != -1:
            iL1 = i
            iL2 = FL2E[fsp[0], (fsp[1] + 1) % 3]
            iL3 = FL2E[fsp[0], (fsp[1] + 2) % 3]

            # A[iA, iL1] = 1
            # A[iA, iL2] = -1
            # A[iA, iL3] = -1
            # b[iA] = - delta
            A_Row[iSp] = iA
            A_Col[iSp] = iL1
            A_Val[iSp] = 1
            iSp += 1

            A_Row[iSp] = iA
            A_Col[iSp] = iL2
            A_Val[iSp] = -1
            iSp += 1

            A_Row[iSp] = iA
            A_Col[iSp] = iL3
            A_Val[iSp] = -1

            Li = FL[fsp[0], fsp[1]]
            Lj = FL[fsp[0], (fsp[1] + 1) % 3]
            Lk = FL[fsp[0], (fsp[1] + 2) % 3]
            B_Val[iA] = Lj + Lk - Li - delta

            iSp += 1

            iA += 1
            nMoll += 1

            # print("i: ", i, "fs: ", fs, "fsp: ", fsp, "iL1: ", iL1, "iL2: ", iL2, "iL3: ", iL3)

        # A[iA, i] = -1
        # b[iA] = - FL[fs[0], fs[1]]
        A_Row[iSp] = iA
        A_Col[iSp] = i
        A_Val[iSp] = -1
        iSp += 1

        B_Val[iA] = 0

        # print("i: ", i, "fs: ", fs, "fsp: ", fsp, "iL1: ", iL1, "iL2: ", iL2, "iL3: ", iL3, "FL[fs[0], fs[1]]: ", FL[fs[0], fs[1]])

        iA += 1

    # slice A and b to remove extra rows based on iA, iSp
    A_Row = A_Row[:iSp]
    A_Col = A_Col[:iSp]
    A_Val = A_Val[:iSp]
    B_Val = B_Val[:iA]

    # Now we can construct the sparse matrix
    A = cvx.spmatrix(A_Val, A_Row, A_Col)
    b = cvx.matrix(B_Val)

    #np.savetxt("A.txt", A)

    # solve
    #print("C:", C.size, "q:", q.size, "A:", A.size, "b:", b.size)
    Eps = cvx.solvers.qp(C, q, A, b)
    Eps = np.array(Eps["x"]).transpose()[0]
    #print(np.shape(Eps), Eps)

    # update FL
    FS = (E2FL[:,0], E2FL[:,1])
    FL[FS] += Eps[FL2E[FS]]

    FSP = G[FS][((G[FS][:,0] != -1) & (G[FS][:,1] != -1)), :]
    FSP = (FSP[:, 0], FSP[:, 1])
    FL[FSP] += Eps[FL2E[FSP]]

    nMoll = nMoll // 3

    #assert CheckInequalityGlobal(FL, delta)
    return FL, nMoll


# def IntrinsicMollification_Global_Optimization_Manhattan_CVX(FL, G, delta = 1e-4):
#     # we want to minimize the Manhattan distance between the original and new edge lengths
#     # we can do this by minimizing the sum of the absolute values of the differences, which is a linear program

#     # we need to build E2FL & FL2E, initially set to -1
#     E2FL, FL2E = BuildFaceAndEdgeMaps(FL, G, delta)

#     cvx.solvers.options['show_progress'] = False

#     # min C'x
#     # np.ones(np.shape(E2FL)[0])
#     C = cvx.matrix(np.ones(np.shape(E2FL)[0]))

#     # s.t. Ax <= b
#     # a + b >= c + delta, b + c >= a + delta, c + a >= b + delta, a>=a_0, b>=b_0, c>=c_0
#     # rewrite 1<->3 as: -a - b + c <= -delta, -b - c + a <= -delta, -c - a + b <= -delta

#     # for cvxopt, we need to store 3 arrays: val, row, col and then construct the sparse matrix later
#     A_Val = np.zeros(np.shape(E2FL)[0] * 7, dtype=float) # atmost 7 non-zero entries per edge (3 for each face, 1 for the edge itself)
#     A_Row = np.zeros(np.shape(E2FL)[0] * 7, dtype=int)
#     A_Col = np.zeros(np.shape(E2FL)[0] * 7, dtype=int)
#     B_Val = np.zeros(np.shape(E2FL)[0] * 3, dtype=float)

#     #A = sp.sparse.lil_matrix((np.shape(E2FL)[0] * 3, np.shape(E2FL)[0]), dtype=float)
#     #b = np.zeros(np.shape(E2FL)[0] * 3, dtype=float)

#     iA = 0
#     iSp = 0
#     nMoll = 0

#     # s.t.

#     for i in range(len(E2FL)):
#         fs = (E2FL[i][0], E2FL[i][1])
#         fsp = tuple(G[fs])

#         iL1 = i
#         iL2 = FL2E[fs[0], (fs[1] + 1) % 3]
#         iL3 = FL2E[fs[0], (fs[1] + 2) % 3]

#         # A[iA, iL1] = 1
#         # A[iA, iL2] = -1
#         # A[iA, iL3] = -1
#         # B[iA] = Lj + Lk - Li - delta
#         A_Row[iSp] = iA
#         A_Col[iSp] = iL1
#         A_Val[iSp] = 1
#         iSp += 1

#         A_Row[iSp] = iA
#         A_Col[iSp] = iL2
#         A_Val[iSp] = -1
#         iSp += 1

#         A_Row[iSp] = iA
#         A_Col[iSp] = iL3
#         A_Val[iSp] = -1

#         Li = FL[fs[0], fs[1]]
#         Lj = FL[fs[0], (fs[1] + 1) % 3]
#         Lk = FL[fs[0], (fs[1] + 2) % 3]
#         B_Val[iA] = Lj + Lk - Li - delta

#         iSp += 1

#         iA += 1
#         nMoll += 1

#         # print("i: ", i, "fs: ", fs, "fsp: ", fsp, "iL1: ", iL1, "iL2: ", iL2, "iL3: ", iL3)

#         if fsp[0] != -1 and fsp[1] != -1:
#             iL1 = i
#             iL2 = FL2E[fsp[0], (fsp[1] + 1) % 3]
#             iL3 = FL2E[fsp[0], (fsp[1] + 2) % 3]

#             # A[iA, iL1] = 1
#             # A[iA, iL2] = -1
#             # A[iA, iL3] = -1
#             # b[iA] = - delta
#             A_Row[iSp] = iA
#             A_Col[iSp] = iL1
#             A_Val[iSp] = 1
#             iSp += 1

#             A_Row[iSp] = iA
#             A_Col[iSp] = iL2
#             A_Val[iSp] = -1
#             iSp += 1

#             A_Row[iSp] = iA
#             A_Col[iSp] = iL3
#             A_Val[iSp] = -1

#             Li = FL[fsp[0], fsp[1]]
#             Lj = FL[fsp[0], (fsp[1] + 1) % 3]
#             Lk = FL[fsp[0], (fsp[1] + 2) % 3]
#             B_Val[iA] = Lj + Lk - Li - delta

#             iSp += 1

#             iA += 1
#             nMoll += 1

#             # print("i: ", i, "fs: ", fs, "fsp: ", fsp, "iL1: ", iL1, "iL2: ", iL2, "iL3: ", iL3)

#         # A[iA, i] = -1
#         # b[iA] = - FL[fs[0], fs[1]]
#         A_Row[iSp] = iA
#         A_Col[iSp] = i
#         A_Val[iSp] = -1
#         iSp += 1

#         B_Val[iA] = 0

#         # print("i: ", i, "fs: ", fs, "fsp: ", fsp, "iL1: ", iL1, "iL2: ", iL2, "iL3: ", iL3, "FL[fs[0], fs[1]]: ", FL[fs[0], fs[1]])

#         iA += 1

#     # slice A and b to remove extra rows based on iA, iSp
#     A_Row = A_Row[:iSp]
#     A_Col = A_Col[:iSp]
#     A_Val = A_Val[:iSp]
#     B_Val = B_Val[:iA]

#     # Now we can construct the sparse matrix
#     A = cvx.spmatrix(A_Val, A_Row, A_Col)
#     b = cvx.matrix(B_Val)

#     #np.savetxt("A.txt", A)

#     # solve
#     #print("C:", C.size, "A:", A.size, "b:", b.size)
#     Eps = cvx.solvers.lp(C, A, b)
#     Eps = np.array(Eps["x"]).transpose()[0]
#     #print(np.shape(Eps), Eps)

#     # update FL
#     FS = (E2FL[:,0], E2FL[:,1])
#     FL[FS] += Eps[FL2E[FS]]

#     FSP = G[FS][((G[FS][:,0] != -1) & (G[FS][:,1] != -1)), :]
#     FSP = (FSP[:, 0], FSP[:, 1])
#     FL[FSP] += Eps[FL2E[FSP]]

#     nMoll = nMoll // 3

#     #assert CheckInequalityGlobal(FL, delta)
#     return FL, nMoll



def IntrinsicMollification_Local_LocalLeastEuclidean(L, delta = 1e-4):
    # we want to minimize the Euclidean distance between the original and new edge lengths
    # we can do this by minimizing the sum of the squares of the differences, which is a quadratic program
    cvx.solvers.options['show_progress'] = False

    # min 1/2 x'Px + q'x
    # we need to minimize 1/2 (a - a_0)^2 + (b - b_0)^2 + (c - c_0)^2
    # x = [a - a_0, b - b_0, c - c_0] = [a, b, c] - [a_0, b_0, c_0] = [u, v, w]
    P = cvx.matrix(np.eye(3, dtype=float)) # 3x3 identity matrix

    q = cvx.matrix(np.zeros((3,1), dtype=float)) # 3x1 zero vector

    # s.t. Gx <= h
    # a_0 + u + b_0 + v >= c_0 + w + delta, b_0 + v + c_0 + w >= a_0 + u + delta, c_0 + w + a_0 + u >= b_0 + v + delta, u>=0, v>=0, w>=0
    # rewrite 1<->3 as: - u - v + w <= a_0 + b_0 - c_0 - delta and so on
    G = cvx.matrix(np.array(
                    [[-1., -1., 1.],
                    [1., -1., -1.],
                    [-1., 1., -1.],
                    [-1., 0., 0.],
                    [0., -1., 0.],
                    [0., 0., -1.]], dtype=float))

    h = cvx.matrix(np.array([0, 0, 0, 0, 0, 0], dtype=float))

    nMoll = 0

    for i in range(len(L)):
        # see if the triangle is already good
        if CheckInequalityLocal(L[i], delta):
            continue

        #print(L[i])
        h[0] = L[i][0] + L[i][1] - L[i][2] - delta
        h[1] = - L[i][0] + L[i][1] + L[i][2] - delta
        h[2] = L[i][0] - L[i][1] + L[i][2] - delta

        # solve
        eps = cvx.solvers.qp(P, q, G, h)
        #print(np.array(eps["x"]).transpose(), L + np.array(eps["x"]).transpose())
        L[i] = L[i] + np.array(eps["x"]).transpose()

        nMoll += 1

    #assert CheckInequalityGlobal(L, delta)

    return L , nMoll


'''
    Parameters:
        V: List of coordinates vertices, a 1D array of length N
        F: List of faces, a N x 3 matrix where each entry is a indexed vertex
        delta: such that l_ij + l_jk > l_ki + delta

    Returns:
        L: columns correspond to original edges lengths [1,2],[2,0],[0,1]
        eps = max_T max(0, delta - l_ij - l_jk - l_ki )
        newL: new edges lengths in the same shape as L
'''

def IntrinsicMollificationConstant(V, F, delta = 1e-4):
    L = igl.edge_lengths(V, F)         # columns correspond to edges lengths [1,2],[2,0],[0,1]
    eps = 0.0

    # replaced the loop above with np operations (vectorized, so much faster)
    eps = max(np.max( [delta + L[:,0] - L[:,1] - L[:,2], delta - L[:,0] + L[:,1] - L[:,2], delta - L[:,0] - L[:,1] + L[:,2] ]  ), 0)

    newL = eps + L
    return L, eps, newL
