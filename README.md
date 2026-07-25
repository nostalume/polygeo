# PolyGeo

PolyGeo is a Python 3.14 reading implementation of discrete differential geometry and discrete exterior calculus.

## Current status

The first executable slice is implemented: an arbitrary-dimensional simplicial core with canonical oriented bases, boundary matrices, simplex subsets, constrained typestate methods, cochain spaces, and generic forms. Geometry, metric DEC, solvers, surface algorithms, and mesh loading remain planned.

Previous topology/surface/dual owner-chain code was rejected and removed. No compatibility with that API is promised; later slices still require explicit approval.

## Design direction

The approved design is based on:

- arbitrary-dimensional simplicial complexes with runtime dimension;
- PEP 695 generic types without explicit `TypeVar` declarations;
- constrained typestate methods for verified topology capabilities;
- cochain spaces that bind basis, runtime degree, and complex identity;
- generic forms parameterized by complex, degree, and semantics;
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
