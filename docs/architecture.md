# PolyGeo Architecture

## Scope

This document describes only the implementation in `src/polygeo`. PolyGeo is an experimental Python 3.14 library for finite simplicial topology, Euclidean geometry, discrete exterior calculus, sparse problem assembly, certified numerical solves, and selected triangle-surface algorithms.

The implemented data flow is:

```text
maximal simplices or optional triangular mesh
  -> Complex refinements
  -> CochainSpace / CochainSubspace and Geometry
  -> LinearMap and AssembledSystem
  -> prepared numerical behavior
  -> forms, certified results, or optional Plotly figures
```

## Components and ownership

| Component | Owner | Responsibility |
|---|---|---|
| Public composition and mesh input | `src/polygeo/__init__.py` | Re-exports the supported root API and implements optional `load_surface()` with a local Trimesh import. |
| Topology and coefficients | `simplicial.py` | Canonical simplices and orientations, boundary matrices, verified refinements, subsets, topological boundary evidence, cochain spaces/subspaces, and forms. |
| Euclidean geometry | `geometry.py` | Admits positions, owns complete primal measures, and computes signed circumcentric dual measures for the exact complex. |
| Cochain maps and metric DEC | `operators.py` | Exact-space `LinearMap`, exterior derivative, restriction/extension, subordinate dual spaces, Hodge star, pairing, codifferential, and Hodge Laplacian. |
| Problem assembly | `systems.py` | Typed operator/right-hand-side systems and canonical-boundary Dirichlet elimination, reconstruction, and solve admission. |
| Numerical behavior | `solvers.py` | Prepared sparse direct factorization, prepared pivoted-QR least squares, residual checks, and stable numerical failures. |
| Dimension-general algorithms | `algorithms.py` | Real homology and periods, positive Hodge admission, Poisson assembly, mean-zero gauge, Hodge decomposition, harmonic extension, and certified result products. |
| Triangle-surface algorithms | `surface.py` | The single surface owner: disk evidence, surface vectors and curvature/flow operators, deterministic triangle frames, geometry-bound SO(2) transport, exact integral dual generators, holonomy/integrability admission, and face direction fields. |
| Optional plotting output | `plotting.py` | Lazy Plotly adapters for geometry, cochains, surface vectors, and homology cycles; it owns presentation only. |
| Exact binary64 helpers | `numerics.py` | Private lattice arithmetic used by assembly, residual, and representability checks. |

## Topology and refinement

`Complex.from_maximal_simplices()` owns canonical bases in every represented degree and the input orientation of maximal simplices. `boundary_matrix(k)` returns the oriented sparse boundary in those bases.

Refinement methods verify topology or state and return another immutable `Complex` view sharing the same private simplicial data. The landed refinements are codimension-one regularity, triangle-manifold topology, orientation, with/without-boundary classification, and connectivity. Boundary classification depends on package-private evidence created during regularity admission; `topological_boundary()` returns its closure-complete subset in parent basis order.

Runtime dimension remains an integer. Python 3.14 PEP 695 parameters express verified state and source/target relationships, while runtime checks preserve the identity of each actual complex.

## Cochains, forms, and maps

A `CochainSpace` binds one exact complex, degree, and canonical simplex ordering. A `CochainSubspace` additionally binds strict parent indices. A `Form` can only be admitted against its exact coefficient space and carries explicit field semantics.

`LinearMap` owns exact source and target spaces plus an admitted finite CSR matrix. Application preserves form semantics; composition requires the exact intermediate space. `exterior_derivative()` accepts explicit adjacent spaces because Python cannot express arbitrary degree arithmetic in its type system. Dual cochain spaces are subordinate views bound to one exact geometry and primal space, not independent dual-complex owners.

## Geometry and metric DEC

`Geometry` is complete when construction returns: positions are finite, ambient dimension is sufficient, every represented simplex is nondegenerate, and every required primal measure is finite and representable. Primal measures are stored in canonical degree order. Signed circumcentric dual measures are computed without persistent dual state and retain negative non-Delaunay contributions.

Metric DEC in `operators.py` combines exact cochain spaces with geometry to build Hodge stars, weighted pairings, codifferentials, and Hodge Laplacians. `PositiveHodgeMetric` in `algorithms.py` admits represented positive Hodge weights across all degrees before algorithms that require a positive metric can run.

## Assembly and numerical behavior

`AssembledSystem` separates mathematical assembly from solving. `eliminate_dirichlet()` accepts an endomorphism on a regular complex, verifies that the prescribed subspace lies on the canonical topological boundary, forms the reduced interior system, and preserves exact reconstruction data.

`prepare_direct()` factors a finite square sparse map and can solve repeated right-hand sides. `prepare_least_squares()` column-scales a full-column-rank rectangular map, uses economic pivoted QR, gates rank and conditioning, and can also solve repeated right-hand sides. Returned solutions include residual evidence in the equation space. Backend exceptions are converted to stable `NumericalError` messages; backend text is not part of the domain contract.

## Current algorithms

`algorithms.py` implements bounded exact-rational real homology representatives, periods of forms on those representatives, scalar Poisson assembly, compatible weighted mean-zero Poisson admission and solving, positive-metric Hodge decomposition in any represented degree, and harmonic extension from canonical boundary values.

`surface.py` is the only owner of triangle-surface behavior. It implements exact disk admission; geometry-bound face and vertex vector values; face unit normals; uniform, tip-angle-weighted, and sphere-inscribed vertex normals; surface-area and enclosed-volume gradients; mean-curvature vectors; integrated interior/boundary angle-defect curvature; and one implicit mean-curvature-flow step using a frozen admitted metric. The flow result contains a completely readmitted target geometry and explicit evidence.

For closed connected oriented triangle surfaces embedded in `R3`, the same owner constructs deterministic right-handed face frames and canonical dual edges, composes Levi-Civita unfolding transport with retained lifted deviation angles as unit-complex SO(2) products, and constructs deterministic primitive integral tree-cotree dual generators. `connection_holonomy()` reports separate absolute circular errors for vertex dual cells and noncontractible generators. `admit_integrable_connection()` is the only factory for the exact-connection-bound capability accepted by `integrate_direction_field()`; the capability certifies represented circular and crossing consistency within a fixed mesh-scaled binary64 limit rather than exact real-arithmetic path independence. The certified field output retains wrapped face phases, ambient unit tangent vectors, anchor provenance, and crossing-consistency evidence. These products are not ordinary cochains.

## Optional input and output boundaries

The root `load_surface()` function is the optional mesh-input boundary. It imports Trimesh only when called, requires exactly one triangular mesh with finite positions, and returns an owned, unrefined `Geometry`. Missing optional support and rejected mesh payloads raise `MeshError` without exposing backend wording.

`plotting.py` is the optional output boundary. Plotly is imported inside each plotting call, and the adapters return ordinary figures without storing presentation state on mathematical values. They verify exact geometry ownership for forms, vector fields, and homology bases. Intrinsic dimensions zero through two are rendered; ambient dimensions above three require an explicit two- or three-axis projection.

## Invariants

- Identity is exact, not structural: geometry, spaces, forms, maps, metrics, subspaces, algorithm capabilities, connections, dual-cycle bases, and plots must refer to the same runtime owner where required. Sharing a generic type or equal arrays is insufficient.
- Public array and sparse-matrix accessors return caller-owned copies. Internal NumPy arrays are owned and read-only; admitted values do not expose mutable caches.
- Refinement never mutates its source and preserves the private simplicial-data identity. Construction and algorithm factories either return complete values or raise.
- Map shape, dtype, finiteness, source/target spaces, degree adjacency, semantics, and represented metric weights are checked at their admission boundaries.
- Domain failures use the boundary-specific families `SimplicialError`, `GeometryError`, `OperatorError`, `SystemError`, `NumericalError`, `AlgorithmError`, `SurfaceError`, `MeshError`, and `PlotError`. Numerical and optional-backend details are not promoted into stable error text.
- Prepared factorizations live only in explicit prepared-solver objects; complexes, geometry, forms, maps, and result values retain no hidden factorization.

## Verification

Runtime tests cover topology laws, boundary extraction, immutable ownership, geometry across extreme scales, DEC adjoint and complex laws, assembly/reconstruction, solver residuals and failure closure, homology/Hodge/Poisson/harmonic algorithms, normals/curvature/flow, surface connection frames and inverse transport, exact tree-cotree generators, local and generator holonomy rejection, capability-gated direction fields, plotting, mesh input, examples, and installed distributions. Typing fixtures exercise accepted and rejected relationships, including rejection of ordinary forms and descriptive evidence at connection capability boundaries.

The repository checks are:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check --error-on-warning .
uv run pytest -q
uv build
```

## Current limitations

- Python 3.14 is required, and the package is experimental.
- Mesh input accepts one triangular Trimesh payload; other mesh families and scene payloads are rejected.
- The numerical layer provides sparse direct and full-column-rank least-squares preparation, not an iterative solver API.
- Boundary assembly implements essential Dirichlet elimination; natural Neumann and Robin assembly are not implemented.
- Plotting is limited to intrinsic dimensions zero through two and requires explicit projection above ambient dimension three.
- Surface connection and direction-field support is restricted to closed connected oriented triangle manifolds embedded in ambient dimension three. It does not implement prescribed singularity indices, lifted singularity solves, boundary-surface transport, or higher-rank structure groups.
