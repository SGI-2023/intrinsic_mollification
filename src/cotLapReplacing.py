import igl
import trimesh

import meshplot as mp

from tqdm import tqdm
from scipy.sparse.linalg import spsolve
from cotanLaplace import cotanLaplace
from Mollification import IntrinsicMollification

    
[V, F] = igl.read_triangle_mesh("../data/bunny.obj")

[E, eps, newL] = IntrinsicMollification(V, F)

l = cotanLaplace(F, E, len(V))
vs = [V]

print('[*] Laplacian smoothing...')
for i in tqdm(range(10)):
    m = igl.massmatrix(V,F,igl.MASSMATRIX_TYPE_BARYCENTRIC)
    V = spsolve(m - 0.001 * l, m.dot(V))
    vs.append(V)

mp.plot(vs[0], F)
mp.plot(vs[4], F)
mp.plot(vs[-1], F)

