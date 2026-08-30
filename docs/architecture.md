# PolyGeo Architecture

## Scope and data flow

PolyGeo is a mixed Rust/Python implementation of finite simplicial topology,
exact chain algebra, discrete exterior calculus, bounded numerical problems,
and triangle-surface geometry.

```text
input arrays or optional mesh
  -> native topology owner
  -> exact Z/Q spaces or binary64 spaces
  -> realization and metric
  -> operator or problem
  -> prepared problem + workspace
  -> native result
  -> explicit NumPy/SciPy/Plotly copy
```

Mathematical and computational roles are separate semantic layers, not
parallel data structures. Full and selected bases are both `Binary64Space`;
chain/cochain values are both `Binary64Element`; every represented binary64
map is `LinearOperator`. Runtime names `Binary64Chain` and `Binary64Cochain` are
aliases of the same element class.

## Ownership

| Component | Responsibility |
|---|---|
| `rust/polygeo-core` | Mathematical authority: topology, exact algebra, realizations, binary64 spaces and maps, problems, solvers, and surface algorithms. |
| `rust/polygeo-py` | Direct native Python carriers, including topology/subset admission, classified failures, GIL release, and explicit copied projections. |
| root namespace and native stub | Direct exact-algebra and homology exports with their generic static relationships; no runtime alias modules. |
| native `HalfedgeSurface` / `SurfaceCorrespondence` | Halfedge admission, topology, conversion witnesses, and fresh caller-owned CSR projections. |
| `mesh.py` | Lazy Trimesh input effect; it immediately admits copied topology and realization owners. |
| `plotting.py` | Lazy Plotly snapshot adapter; it owns presentation only. |
| `__init__.py` | Explicit public exports, including identity re-exports of the two optional leaves. |

Native handles retain only the owners required by their mathematical identity;
a surface correspondence retains its exact public source and target objects.
Python input is borrowed for admission and copied into native storage. Returned
NumPy and SciPy objects are caller-owned snapshots.

## Topology and exact algebra

`Complex.from_maximal_simplices()` admits one immutable native topology owner.
Refinement methods verify a capability and return that same native object.
Evidence and deterministic rejections are cached in the native owner.

Exact algebra uses one generic native `ChainComplex`, `Space`, `Element`, and
`LinearMap` family over integer or rational coefficients and chain/cochain
variance. Exact values have no implicit binary64 or array conversion.
`to_python_copy()` and checked `to_scipy_int64_copy()` are explicit copies.
Requested-degree integral homology is prepared under an immutable resource
limit and retains its exact owner.

## Binary64 layer

`complex.binary64_chain_space(k)` and
`complex.binary64_cochain_space(k)` derive direct native spaces. Passing
`indices=` selects a canonical sub-basis without constructing a different
space class. A space admits a `float64` NumPy vector or realizes a compatible
exact integral element. Values expose only `coefficients_numpy_copy()`.

Spaces construct identity and exterior-derivative maps. Metrics construct
Riesz, inverse-Riesz, codifferential, and Laplacian maps. Operators retain their
native source and target identities and reject foreign values.

## Geometry, metrics, and problems

`EuclideanRealization` owns finite positions and admitted primal geometry;
`Geometry` is an alias of that class. Projection names state allocation:
`positions_numpy_copy()`, `primal_measures_numpy_copy()`, and
`dual_measures_numpy_copy()`.

`realization.positive_metric()` admits represented positive Hodge weights and
constructs metric maps or one of the native problem variants. Every problem
follows the same lifecycle:

```text
problem.prepare(limits, cancellation)
  -> prepared.workspace_for(problem)
  -> prepared.solve(problem, workspace, limits, cancellation)
```

Preparation and solving release the GIL. Storage and work limits are explicit;
cancellation is cooperative. A result is published only after a complete
solve. Flow results contain a newly admitted target realization and never
mutate their source.

## Triangle surfaces

`TriangleSurface.admit(realization)` is the surface authority. Every vertex or
face vector field uses the single `EntityVectors` carrier; `VertexVectors` and
`FaceVectors` are aliases. Normals, gradients, curvature, frames, connections,
dual-cycle evidence, holonomy, integrability, and direction fields retain the
same native surface/realization spine. Array observations use explicit
`*_numpy_copy()` names.

## Optional effects

`polygeo.mesh.load_surface()` imports Trimesh only when called and returns an
owned native realization. Plotting derives typed declarative traces from copied
position, coefficient, or vector snapshots, then constructs one Plotly figure
at one lazy effect boundary. Root and leaf names have the same identity.
Optional libraries never own or alter mathematical state.

## Invariants

- Identity is nominal and owner-bound; equal arrays do not imply compatible
  spaces, maps, metrics, or problems.
- Mathematical roles do not create duplicate runtime carriers.
- Public dense and sparse projections are explicit caller-owned copies.
- Admission and result publication are atomic.
- Domain failures are classified by the relevant topology, exact algebra,
  geometry, operator, problem, solve, surface, mesh, or plotting error family.
- Prepared numerical state exists only in `PreparedProblem` and
  `SolveWorkspace`, never in topology, realization, or element values.
