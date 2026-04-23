# PolyGeo Design

## Language

**Python** is the recommended primary language for this project.

### Rationale

- Numpy/scipy match C++ Eigen for most DDG tasks
- Dense + sparse solve in one line with scipy
- Polyscope provides instant visualization
- Jupyter notebooks enable inspecting mesh state mid-algorithm
- Faster iteration than C++ (no compilation overhead)
- gpytoolbox and robust_laplacian provide reference implementations

### Alternatives

- **Rust**: Good for performance-critical inner loops, but sparse solver ecosystem less mature. Consider later for production pipelines via PyO3.
- **C++**: Original book's language but tooling overhead (CMake, libigl, Eigen) slows learning.

**Pragmatic hybrid**: Write algorithms in Python first; port hot inner loops to Rust only if real-time performance needed.

## Dependencies

### Core

- `numpy`: Dense arrays for vertex positions, face normals, per-vertex data
- `scipy.sparse`: CSR/COO sparse matrices for DEC operators; `spsolve`/`splu` for Poisson/flow systems
- `polyscope`: Interactive visualization for meshes, scalar/vector fields, geodesics

### Geometry Tooling

- `gpytoolbox`: Geometry processing utilities (cotan-Laplace, mass matrix, mesh I/O, geodesics)
- `trimesh`: Lightweight mesh loading for test cases

### Optional

- `scikit-sparse`: SuiteSparse wrapper, 3-5× faster sparse Cholesky for meshes >100k vertices
- `networkx`: Graph algorithms for homology, fundamental polygon, dual spanning trees
- `jax`: Auto-diff + JIT for differentiating through geometric quantities

```bash
pip install numpy scipy polyscope gpytoolbox robust-laplacian igl trimesh
# optional:
pip install scikit-sparse
```

## Architecture — Four Layers

### Layer 0: Mesh Representation

Halfedge data structure as arrays, not pointer-chasing structures.

**Data structures:**

- `V : (n_v, 3) float64` — vertex positions
- `F : (n_f, 3) int32` — triangle vertex indices
- `E : (n_e, 2) int32` — unique undirected edge pairs, sorted
- `edge_idx : dict[(a,b) → int]` — edge index lookup

Built from mesh I/O (`gpytoolbox.read_mesh()` or `igl.read_triangle_mesh()`).

### Layer 1: Discrete Exterior Calculus Operators

All DEC operators are sparse matrices. Build once; compose for all algorithms.

- `d0 : (n_e × n_v)` — signed vertex-edge incidence (−1 tail, +1 head)
- `d1 : (n_f × n_e)` — signed edge-face incidence (encodes ∂)
- `star0 : (n_v × n_v)` — diagonal, dual vertex areas
- `star1 : (n_e × n_e)` — diagonal, cotan weights per edge
- `star2 : (n_f × n_f)` — diagonal, face areas

> Every algorithm in the book reduces to matrix arithmetic. Cotan-Laplace: `d0.T @ star1 @ d0`; Co-differential: `star0 @ d0.T @ star1`.

### Layer 2: Laplace-Beltrami & Solvers

Cotan-Laplace `L` and lumped mass matrix `M` power all applications.

- `L = d0.T @ star1 @ d0` or `robust_laplacian.mesh_laplacian(V, F)`
- `M[i,i] = (1/3) × Σ area(triangles incident on vertex i)`
- **Poisson**: `spsolve(L, M @ rho)` with one pinned DOF
- **Implicit flow**: `spsolve(M - h*L, M @ f0)` per time step
- **Eigenvalue**: `scipy.sparse.linalg.eigsh(L, M=M, k=2, which='SM')`

**Solver choice:** `spsolve` (LU) fine to ~50k vertices. Larger meshes: `sksparse.cholmod.cholesky(L)` for reuse.

### Layer 3: Applications

Thin wrappers; DEC stack does heavy lifting.

- **Gaussian curvature**: per-vertex angle defect (no solve)
- **Mean curvature normals**: `L @ V / (2 * M.diagonal()[:,None])`
- **Implicit flow**: one sparse solve per step
- **Conformal parameterization**: smallest eigenvector of `L` (ARPACK)
- **Hodge decomposition**: two Poisson solves on `d0`, `d1`
- **Heat geodesics**: solve heat → normalize grad → Poisson (three lines)
- **Vector field design**: connection Laplacian on complex 1-forms

## Minimal Skeleton

```python
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import polyscope as ps
import gpytoolbox as gpy
import robust_laplacian

# Layer 0: load mesh
V, F = gpy.read_mesh("bunny.obj")
n_v, n_f = len(V), len(F)

# Build unique undirected edges
edges = set()
for tri in F:
    for i in range(3):
        edges.add(tuple(sorted([tri[i], tri[(i+1)%3]])))
edges = np.array(sorted(edges))
n_e = len(edges)
edge_idx = {tuple(e): i for i, e in enumerate(edges)}

# Layer 1: DEC operators
# d0: signed vertex→edge incidence (n_e × n_v)
rows, cols, vals = [], [], []
for i, (a, b) in enumerate(edges):
    rows += [i, i]; cols += [a, b]; vals += [-1, 1]
d0 = coo_matrix((vals, (rows, cols)), shape=(n_e, n_v)).tocsr()

# d1: signed edge→face incidence (n_f × n_e)
rows, cols, vals = [], [], []
for f_idx, tri in enumerate(F):
    for i in range(3):
        a, b = tri[i], tri[(i+1)%3]
        eidx = edge_idx[tuple(sorted([a,b]))]
        sign = 1 if a < b else -1
        rows.append(f_idx); cols.append(eidx); vals.append(sign)
d1 = coo_matrix((vals, (rows, cols)), shape=(n_f, n_e)).tocsr()

# Layer 2: cotan-Laplace
L, M = robust_laplacian.mesh_laplacian(V, F)

# Layer 3: Poisson solve
rho = np.zeros(n_v); rho[0] = 1.0; rho[-1] = -1.0
A = L.copy().tolil(); A[0, :] = 0; A[0, 0] = 1
b = M @ rho; b[0] = 0
phi = spsolve(A.tocsr(), b)

# Visualize
ps.init()
mesh = ps.register_surface_mesh("mesh", V, F)
mesh.add_scalar_quantity("potential", phi, enabled=True)
ps.show()
```

**Extend this skeleton:**

- Mean curvature flow: replace `rho` with vertex positions, add time step
- Heat geodesics: two more solves
- Parameterization: eigenvalue problem on same `L`
