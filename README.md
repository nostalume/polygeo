# PolyGeo

PolyGeo is a Python 3.14 reading implementation of discrete differential geometry and discrete exterior calculus.

## Current status

The implemented core includes arbitrary-dimensional simplicial topology, complete Euclidean geometry, and signed metric DEC operators: canonical oriented bases, boundary matrices, simplex subsets, constrained typestate methods, dimension-general boundary-regular admission and extraction, parent-retaining cochain subspaces, restriction and zero extension, primal and subordinate dual cochain spaces, generic forms, typed linear maps, owned positions, scale-safe primal measures, signed circumcentric dual measures, exterior derivatives, Hodge stars, weighted pairings, codifferentials, and Hodge Laplacians. Solvers, assembled boundary-value problems, and higher algorithms remain planned.

Previous topology/surface/dual owner-chain code was rejected and removed. No compatibility with that API is promised; later slices still require explicit approval.

## Design direction

The approved design is based on:

- arbitrary-dimensional simplicial complexes with runtime dimension;
- PEP 695 generic types without explicit `TypeVar` declarations;
- constrained typestate methods for verified topology capabilities;
- cochain spaces that bind basis, runtime degree, and complex identity;
- generic forms parameterized by exact coefficient space and semantics;
- typed linear maps with explicit source and target spaces;
- specific-dimensional algorithms constrained by mathematical capability rather than `SurfaceOneForm`-style subclasses;
- complete construction with no `object.__setattr__`, empty shells, or hidden instance caches;
- a future root-level `load_surface` rather than organizational `api.py` or `io.py` modules.

## Simplicial core

```python
import numpy as np

from polygeo import Complex, ORDINARY_FORM

raw = Complex.from_maximal_simplices(
    np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
)
surface = raw.triangle_manifold().oriented().connected()

one_space = surface.cochain_space(1)
one_form = one_space.form(np.zeros(one_space.size), ORDINARY_FORM)
boundary_2 = surface.boundary_matrix(2)
```

`closed()` is a separate refinement and correctly rejects this disk example.

## General geometry

```python
from polygeo import Geometry

geometry = Geometry.from_positions(
    raw,
    np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    ),
)

edge_lengths = geometry.primal_measures(1)
triangle_areas = geometry.primal_measures(2)
dual_edge_lengths = geometry.dual_measures(1)
```

`Geometry[K]` preserves the exact complex identity and supports any intrinsic dimension representable in its runtime ambient dimension. Both measure methods use the associated primal degree and canonical primal simplex indices: `primal_measures(k)[i]` is $|\sigma_i^k|$, while `dual_measures(k)[i]` is the signed circumcentric $|\star\sigma_i^k|$ of runtime dimension $n-k$. The dual computation is pure and uncached; no measure-only dual owner or duplicated dual complex exists. Triangle normals, corner angles, and cotangents remain separate constrained computations.

## Topological exterior derivative

```python
from polygeo import exterior_derivative

zero_space = raw.cochain_space(0)
one_space = raw.cochain_space(1)
d0 = exterior_derivative(zero_space, one_space)

zero_form = zero_space.form(
    np.arange(zero_space.size, dtype=np.float64),
    ORDINARY_FORM,
)
one_form = d0.apply(zero_form)

two_space = raw.cochain_space(2)
d1 = exterior_derivative(one_space, two_space)
zero_map = d1.compose(d0)
assert zero_map.matrix().nnz == 0
```

`Form[Space, Semantics]` and `LinearMap[SourceSpace, TargetSpace]` retain exact coefficient-space identity. `Form.coefficients()` and `LinearMap.matrix()` return caller-owned representations. Map application preserves field semantics, while composition statically unifies the intermediate space and checks its exact runtime identity. The exterior derivative still accepts explicit adjacent `CochainSpace` values because Python cannot calculate `SourceDegree + 1` at the type level; the chain boundary remains `Complex.boundary_matrix(k)` and the cochain derivative uses its transpose. `hodge_star(geometry, source)` derives a subordinate `DualCochainSpace` that references the exact geometry and primal space without duplicating a `DualComplex`.

## Documentation

1. [`docs/architecture.md`](docs/architecture.md) — target concepts, API shape, Python 3.14 type model, invariants, and forbidden structures.
2. [`docs/roadmap.md`](docs/roadmap.md) — type-system spikes, test-first implementation tasks, review firewall, and verification gates.

## Toolchain

The declared environment uses Python 3.14, `uv`, NumPy, SciPy, pytest, Ruff, and Ty.

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check --error-on-warning .
uv run pytest -q
uv build
```

## License

MIT
