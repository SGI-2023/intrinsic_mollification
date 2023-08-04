import igl
import trimesh

import meshplot as mp

from tqdm import tqdm
from scipy.sparse.linalg import spsolve

from cotanLaplace import cotanLaplace
from Mollification import IntrinsicMollification

    
[v, f] = igl.read_triangle_mesh("../data/bunny.obj")

[E, eps, newL] = IntrinsicMollification(v, f)

l = cotanLaplace(f, E)
vs = [v]


#Laplacian Smoothing
print('[*] Laplacian smoothing...')
for i in tqdm(range(10)):
    m = igl.massmatrix(v,f,igl.MASSMATRIX_TYPE_BARYCENTRIC)
    v = spsolve(m - 0.001 * l, m.dot(v))
    vs.append(v)

# mp.plot(vs[0], f)
# mp.plot(vs[4], f)
# mp.plot(vs[-1], f)

mesh0 = trimesh.Trimesh(vertices= vs[0], faces= f)
mesh1 = trimesh.Trimesh(vertices= vs[4], faces= f)
mesh2 = trimesh.Trimesh(vertices= vs[-1], faces= f)

scene0 = trimesh.Scene([mesh0])
scene1 = trimesh.Scene([mesh1])
scene2 = trimesh.Scene([mesh2])

scene0.show()
scene1.show()
scene2.show()