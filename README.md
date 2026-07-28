# PolyGeo

PolyGeo is a reading implementation of discrete differential geometry and discrete exterior calculus for finite simplicial complexes. It provides typed topology, Euclidean geometry, cochains and DEC operators, numerical assembly, and a focused set of mesh algorithms through the `polygeo` package root.

PolyGeo requires Python 3.14. Version 0.1.0 is experimental: the implemented paths are tested, but the public API and numerical policies are not yet stable.

## Install from source

Install the core development environment with [uv](https://docs.astral.sh/uv/):

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

- `polygeo[mesh]` enables the root `load_surface()` boundary, which uses Trimesh to read one triangular mesh into an unrefined `Geometry`.
- `polygeo[plot]` enables the root plotting functions, which return ordinary Plotly figures. Importing `polygeo` does not import Plotly.

## Quick start

This complete script builds an oriented triangulated disk, admits its Euclidean geometry, computes integrated Gaussian curvature, and checks Gauss–Bonnet:

```python
import math

import numpy as np

from polygeo import Complex, Geometry, gaussian_curvature_measure

faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
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
curvature = gaussian_curvature_measure(geometry)

assert math.isclose(
    math.fsum(curvature.coefficients()),
    2.0 * math.pi,
    rel_tol=0.0,
    abs_tol=1e-12,
)
print(curvature.coefficients())
```

Save it as `quick_start.py` and run `uv run python quick_start.py`.

## Implemented capabilities

| Group | Current implementation |
|---|---|
| Topology and cochains | Arbitrary-dimensional canonical simplex bases, oriented boundary matrices, subsets and topological boundary extraction, regular/triangle-manifold/oriented/boundary/connected refinements, cochain spaces and subspaces, forms, restriction, and zero extension. |
| Geometry and metric DEC | Complete Euclidean geometry, scale-safe primal measures, signed circumcentric dual measures, exterior derivative, Hodge star, weighted pairing, codifferential, Hodge Laplacian, and represented-positive Hodge metric admission. |
| Assembly and numerics | Typed assembled systems, canonical-boundary Dirichlet elimination and reconstruction, prepared sparse direct solves, prepared full-column-rank least squares, and residual evidence. |
| General algorithms | Exact real homology representatives and periods, scalar Poisson assembly, compatible mean-zero Poisson solving, all-degree Hodge decomposition, and harmonic extension from canonical boundary values. |
| Triangle surfaces | Disk admission, face and vertex normal constructions, area and volume gradients, mean-curvature vectors, integrated Gaussian curvature, one frozen-metric implicit flow step, deterministic face frames, geometry-bound SO(2) connection transport, exact integral dual generators, local/global holonomy evidence, factory-only integrability, and ambient face direction fields. |
| Optional boundaries | Root Trimesh surface input plus Plotly output for geometry, cochains, surface vectors, and retained homology cycles. |

## Examples

The executable Marimo studies in [`examples/`](examples/) cover curvature, Poisson, harmonic extension, mean-curvature flow, homology and periods, Hodge decomposition, and connection and holonomy. See the [examples guide](examples/README.md).

```bash
uv run --extra mesh --extra plot marimo edit examples/curvature.py
uv run --extra mesh --extra plot marimo check --strict examples/*.py
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for current component ownership, data flow, invariants, optional boundaries, and limitations.

## Development checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check --error-on-warning .
uv run pytest -q
uv build
```

## License

PolyGeo is licensed under the MIT License.
