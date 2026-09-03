# PolyGeo Architecture

## Scope and data flow

PolyGeo is a mixed Rust/Python implementation of finite simplicial topology,
exact chain algebra, discrete exterior calculus, bounded numerical problems,
and triangle-surface geometry.

```text
input arrays or optional mesh
  -> native topology owner
  -> exact Z/Q chain spaces or binary64 form spaces
  -> geometry and metric
  -> operator or problem
  -> prepared problem + workspace
  -> native result
  -> explicit NumPy/SciPy/Plotly copy
```

Mathematical and computational roles are separate semantic layers, not
parallel data structures. Full and selected bases are both `form.Space`;
chain/cochain values are both `form.Element`; every represented binary64 map is
`form.Operator`.

## Ownership

| Component | Responsibility |
|---|---|
| `rust/polygeo-core` | Mathematical authority: topology, exact algebra, realizations, binary64 spaces and maps, problems, solvers, and surface algorithms. |
| `rust/polygeo-py` | Direct native Python carriers, including topology/subset admission, classified failures, GIL release, and explicit copied projections. |
| `polygeo` package | Eight contextual module objects only: `topology`, `chain`, `form`, `geometry`, `solve`, `field`, `plot`, and `mesh`. |
| native `HalfedgeSurface` / `ChainIsomorphism` | Halfedge admission, topology, direct ordered conversion witnesses, and fresh caller-owned projections. |
| `mesh.py` | Lazy Trimesh input effect; it immediately admits copied topology and realization owners. |
| `plot.py` | Lazy Plotly snapshot adapter; it owns presentation only. |
| `__init__.py` | Imports the eight contextual modules without re-exporting their members. |

Native handles retain only the owners required by their mathematical identity;
surface conversion returns one ordered chain isomorphism whose source and target
complexes retain their exact owners.
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
Requested-degree integral homology is analyzed under an immutable resource
limit and retains its exact owner.

## Forms

`complex.binary64_chain_space(k)` and
`complex.binary64_cochain_space(k)` derive direct native spaces. Passing
`indices=` selects a canonical sub-basis without constructing a different
space class. A space admits a `float64` NumPy vector or realizes a compatible
exact integral element. Values expose only `coefficients_numpy_copy()`.

Spaces construct identity and exterior-derivative maps. Metrics construct
Riesz, inverse-Riesz, codifferential, and Laplacian maps. Operators retain their
native source and target identities and reject foreign values.

## Geometry, metrics, and problems

`geometry.Geometry` owns finite positions and admitted primal geometry.
Projection names state allocation:
`positions_numpy_copy()`, `primal_measures_numpy_copy()`, and
`dual_measures_numpy_copy()`.

`geometry.metric()` admits represented positive Hodge weights and
constructs metric maps, reusable problem variants, or direct bounded analyses.
Every reusable problem follows the same lifecycle:

```text
problem.prepare(policy, cancellation)
  -> prepared.workspace_for(problem)
  -> prepared.solve(problem, workspace, cancellation)
```

Preparation and solving release the GIL. One-shot harmonic-basis, frozen-flow,
and LSCM computations execute directly instead of publishing unusable preparation
objects. A prepared computation retains its executor, storage, and work policy;
cancellation remains a separate cooperative input. A result is published only
after complete certification. Harmonic bases retain
existing binary64 cochains normalized to exact homology periods; flow and LSCM
results contain newly admitted realizations and never mutate their source.

## Triangle surfaces

`TriangleSurface.admit(geometry)` is the surface authority. Every vertex or
face vector field uses the single support-indexed `geometry.VectorField[Degree]`
carrier; `VertexField` and `FaceField` select degrees zero and two without
retained core tags. Normals, gradients, curvature, frames, connections,
dual-cycle evidence, holonomy, integrability, and direction fields retain the
same native surface/realization spine. A positive runtime symmetry order lifts
Levi-Civita transport and face coordinates into one branch-free power
representation; ordinary, line, and cross fields do not create separate
carriers. Array observations use explicit `*_numpy_copy()` names, and ambient
vectors are requested one explicit local branch at a time.

Singularity observation stays in power coordinates: interior integer charges,
ordered exact boundary turns, and local quantization evidence satisfy
`sum(charges) - sum(boundary_turns) = symmetry_order * Euler(surface)`. Closed
surfaces have no boundary turns. The geometric index remains the rational
interpretation `charge / symmetry_order`, not a second coefficient carrier.

Boundary-aligned fields use the same face carrier and compact interior transport.
A relaxed connection-Dirichlet extension selects one representable phase-lift
sector; a second scalar Dirichlet solve minimizes connection deviation within
that sector. This is not a global minimum over singularity configurations.

Symmetric direction-field synthesis composes those same objects. An exact
degree-zero cochain prescribes power charges, exact dual cycles order integer
generator turns, and period-normalized harmonic one-forms provide global
freedom. One compatible Poisson load and a temporary period solve publish the
existing `field.Direction`; exact charges and compact quantization evidence
remain observable afterward.

## Optional effects

`polygeo.mesh.load_surface()` imports Trimesh only when called and returns an
owned native geometry. Plotting derives typed declarative traces from copied
position, coefficient, or vector snapshots, then constructs one Plotly figure
at one lazy effect boundary. Effects are accessed only through their contextual
modules.
Optional libraries never own or alter mathematical state.

## Invariants

- Identity is nominal and owner-bound; equal arrays do not imply compatible
  spaces, maps, metrics, or problems.
- Mathematical roles do not create duplicate runtime carriers.
- Public dense and sparse projections are explicit caller-owned copies.
- Admission and result publication are atomic.
- Domain failures are classified by the relevant topology, exact algebra,
  geometry, operator, problem, solve, surface, mesh, or plotting error family.
- Prepared numerical state exists only in `solve.Prepared` and
  `solve.Workspace`, never in topology, geometry, or form values.
