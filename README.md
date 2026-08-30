# PolyGeo

PolyGeo is a reading implementation of discrete differential geometry and discrete exterior calculus for finite simplicial complexes. It provides typed topology, Euclidean geometry, cochains and DEC operators, numerical assembly, and a focused set of mesh algorithms through the `polygeo` package root.

PolyGeo requires Python 3.14. Version 0.1.0 is experimental: the implemented paths are tested, but the public API and numerical policies are not yet stable.

## Install from source

Install [Rust 1.97.1](https://www.rust-lang.org/tools/install) and the core
development environment with [uv](https://docs.astral.sh/uv/). The pinned Rust
toolchain is used to compile the private native topology core:

```bash
git clone https://github.com/nostalume/polygeo.git
cd polygeo
uv sync
```

Optional dependencies are separate:

```bash
uv sync --extra mesh
uv sync --extra plot
# or install both
uv sync --extra mesh --extra plot
```

- `polygeo[mesh]` enables `polygeo.mesh.load_surface()` (also re-exported at the root), which uses Trimesh to read one triangular mesh into an unrefined `Geometry`.
- `polygeo[plot]` enables `polygeo.plotting` snapshot functions for realizations, full binary64 forms, selected free homology cycles, and native surface-vector fields (also re-exported at the root). They return ordinary Plotly figures. Importing `polygeo` imports neither optional dependency.

## Quick start

This complete script builds an oriented triangulated disk, admits its Euclidean geometry, computes integrated Gaussian curvature, and checks Gauss–Bonnet:

```python
import math

import numpy as np

from polygeo import Complex, Geometry, TriangleSurface

faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

surface = (
    Complex.from_maximal_simplices(faces)
    .triangle_manifold()
    .oriented()
    .with_boundary()
    .connected()
)
geometry = Geometry.from_positions(surface, positions)
triangle_surface = TriangleSurface.admit(geometry)
curvature = triangle_surface.gaussian_curvature_measure()
coefficients = curvature.coefficients_numpy_copy()

assert math.isclose(
    math.fsum(coefficients),
    2.0 * math.pi,
    rel_tol=0.0,
    abs_tol=1e-12,
)
print(coefficients)
```

Save it as `quick_start.py` and run `uv run python quick_start.py`.

## Implemented capabilities

| Group | Current implementation |
|---|---|
| Topology and binary64 values | Arbitrary-dimensional canonical simplex bases, oriented boundary matrices, subsets and topological boundary extraction, refinements, and one native `Binary64Space`/`Binary64Element`/`LinearOperator` family for full or selected chain and cochain bases. |
| Exact chain algebra | Direct native owner-bound sparse chains and algebraic-dual cochains over Z and Q, explicit scalar extension, checked map composition and duality, bounded explicit CSR materialization, requested-degree integral homology, and checked owner transport through surface chain isomorphisms. |
| Combinatorial surfaces | Immutable orientable halfedge owners, material/exterior face separation, an exact integral chain complex, explicit eligible triangle-complex conversion with checked chain isomorphism, and owner-local component/Euler/genus facts. |
| Geometry and metric DEC | Native `EuclideanRealization` (`Geometry` is the same class), explicit copied projections, primal and signed circumcentric dual measures, exterior derivative, Riesz maps, codifferential, Hodge Laplacian, and positive metric admission. |
| Problems and numerics | Native reusable problem/preparation/workspace carriers for Dirichlet, compatible mean-zero Poisson, Hodge decomposition, harmonic extension, and scalar heat; direct bounded computations for frozen mean-curvature flow, LSCM, and period-normalized harmonic one-form bases; all with cancellation and residual evidence. |
| Triangle surfaces | Disk admission, face and vertex normal constructions, area and volume gradients, mean-curvature vectors, integrated Gaussian curvature, one frozen-metric implicit flow step, deterministic face frames, geometry-bound SO(2) connection transport, exact integral dual generators, local/global holonomy evidence, factory-only integrability, exact singularity observation, and minimum-energy ordinary direction fields with prescribed integer topology. |
| Optional boundaries | Root Trimesh surface input plus Plotly snapshots for geometry, full forms, free homology-cycle selections, and native surface-vector fields. |

Exact integral homology is analyzed explicitly under immutable resource limits.
Analyses, group views, and representatives retain their native owner; Python,
NumPy, SciPy, and binary64 compatibility values are explicit owned projections.

## Examples

The seven executable Marimo studies in [`examples/`](examples/) cover curvature,
Poisson, LSCM, heat-method distance, homology, Hodge decomposition, and connection
holonomy. See the [examples guide](examples/README.md).

## Architecture

See [docs/architecture.md](docs/architecture.md) for current component ownership, data flow, invariants, optional boundaries, and limitations.

## Current limitations

- Python 3.14 and the pinned Rust toolchain are required.
- Mesh input accepts one triangular Trimesh payload.
- The native executor is currently sequential.
- No stable-ABI or free-threaded-Python guarantee is made.
- Surface connections require closed, connected, oriented triangle manifolds
  embedded in three dimensions.

The [verification workflow](.github/workflows/verify.yml) is the canonical
platform, quality, example, and installed-artifact command matrix.

## License

PolyGeo is licensed under the MIT License.
