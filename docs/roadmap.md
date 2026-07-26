# PolyGeo Implementation Tasks

## Status and Authorization

This document is a proposed implementation sequence for [`architecture.md`](architecture.md). It is not authorization to implement it.

Current implemented baseline:

- arbitrary-dimensional simplicial topology, boundary-regular extraction, parent-retaining cochain subspaces, restriction and zero extension, and forms;
- complete arbitrary-dimensional Euclidean geometry, primal simplex measures, and signed circumcentric dual measures;
- typed linear maps, composition, arbitrary-dimensional exterior derivative, subordinate dual cochain identity, signed Hodge star, weighted pairing, codifferential, and Hodge Laplacian;
- typed assembled systems plus true-boundary, endomorphism-only Dirichlet block elimination and exact reconstruction;
- operation-scoped sparse direct factorization, repeated-right-hand-side reuse, and certified residual evidence;
- executable runtime, positive Ty, and negative Ty contracts;
- rejected owner-chain modules remain retired and must not be restored.

Before any task starts, the user must explicitly approve the task or a bounded group of tasks. No commit, push, force-push, reset, restore, or index cleanup is implied by this plan.

## Delivery Rules

Every task must satisfy all of the following:

- less than four hours of intended work;
- less than 500 changed lines; split before crossing the limit;
- RED test or typecheck evidence before production implementation;
- one mathematical concept per task;
- no compatibility shim for the rejected architecture;
- no movement to the next task while the current task has failing focused gates;
- exact review after each typestate or numerical-boundary change;
- no claim of completion from build success alone.

The implementation must use Python 3.14 and `uv`. Source code may use PEP 695 inline type parameters but must not declare `typing.TypeVar` or old-style `Generic` bases.

## Target Physical Surface

Physical layout is intentionally compact. It is an implementation consequence, not the architecture authority.

```text
src/polygeo/
  __init__.py      root exports
  simplicial.py    complete complex data, refinements, cochain spaces and topology
  geometry.py      complete general Euclidean geometry and simplex measures
  operators.py     typed maps and metric DEC operators
  systems.py       assembled linear systems and boundary elimination
  numerics.py      package-private exact binary64 arithmetic
  solvers.py       domain-neutral prepared numerical behavior

tests/
  test_simplicial.py
  test_geometry.py
  test_operators.py
  test_systems.py
  test_solvers.py
  test_typing.py
  typing/             flat positive and negative real-Python Ty fixtures
```

Do not create `api.py`, `io.py`, `errors.py`, `result.py`, `forms.py`, `metric.py`, `dual.py`, or one test file per source module unless a later approved design demonstrates a separate behavior owner. General geometry and executable negative type contracts are approved behavior owners, not organizational mirrors.

## Mandatory Gates

Each implemented task runs its focused test and typecheck commands. A phase exit additionally requires:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check --error-on-warning .
uv run pytest -q
```

When examples exist:

```bash
uv run marimo check <all tracked Marimo studies>
```

Final delivery additionally requires:

```bash
uv lock --check
uv build
```

Then install and smoke-test the wheel and sdist independently in fresh environments. Build success without import and behavior smoke is insufficient.

## T0 — Repository and Type-System Admission

### TYPE-00 — Confirm the implementation baseline

**Purpose:** Prevent implementation on an ambiguous soft-reset/index state.

**Prerequisite:** Explicit user approval for repository-state handling.

**Work:**

1. Record `HEAD`, branch, staged paths, unstaged paths, and untracked paths.
2. Confirm which staged historical files are intentionally retained or discarded.
3. Confirm no deleted production module or test should be restored.
4. Agree on whether implementation changes remain unstaged or replace the staged index.
5. Do not commit or rewrite remote history.

**Exit evidence:** A reported, user-confirmed baseline. No code change.

### TYPE-01 — PEP 695 refinement spike

**Purpose:** Prove the central type transition before building domain values.

**Prerequisite:** `TYPE-00`.

**Work:** Create a disposable typecheck spike outside production code. It models only tiny scalar placeholders and proves:

```text
Complex[BoundaryUnknown, O, C, CodimensionOneRegular]
    .without_boundary()
-> Complex[WithoutBoundary, O, C, CodimensionOneRegular]
```

The spike must test:

- PEP 695 bounded class parameters;
- `Refinement[Source, Target]` behavior;
- `Self`-based `.refine(rule)` inference;
- preservation of unrelated axes;
- rejection of refinement from an invalid source state;
- runtime behavior dispatch through the rule object;
- no runtime inspection of parameterized generics.

**RED cases:** Ty must reject:

- passing a closed-only value before refinement;
- applying `without_boundary()` to an already classified boundary state;
- losing orientation/connectivity/topology axes during transition.

**Hard failure conditions:**

- `Any`;
- opaque `object`;
- `typing.TypeVar`;
- old-style `Generic`;
- `cast` or `type: ignore` required at every transition;
- Ty silently resolves target state to Unknown.

**Result:** Generic `.refine(rule)` inferred `Unknown`. Direct constrained methods such as `connected(self: Complex[B, O, ConnectivityUnknown, T]) -> Complex[B, O, Connected, T]` passed positive and negative Ty probes and are the accepted API.

### TYPE-02 — Generic cochain-space and form spike

**Purpose:** Prove `Complex × degree × semantics` without `SurfaceOneForm`.

**Prerequisite:** `TYPE-01`.

**Work:** Prove:

```text
CochainSpace[K, Literal[1]]
    -> Form[CochainSpace[K, Literal[1]], OrdinaryForm]

CochainSpace[K, Literal[1]]
    -> Form[CochainSpace[K, Literal[1]], SO2Connection]
```

The spike must determine whether Ty preserves `Literal[1]` from a literal space-construction call. If it widens to `int`, define the narrowest honest fixed-degree constructor or refinement. Do not enumerate arbitrary dimensions or degrees.

**RED cases:** Ty must reject:

- degree-zero form where degree one is required;
- ordinary form where connection semantics are required;
- form constructed from coefficients without a cochain space.

**Exit evidence:** Ty diagnostics prove positive and negative cases.

### TYPE-03 — Typed linear-map spike

**Purpose:** Avoid unsupported type-level `degree + 1` arithmetic.

**Prerequisite:** `TYPE-02`.

**Work:** Prove:

```text
LinearMap[SourceSpace, TargetSpace]
    .apply(Form[SourceSpace, S])
    -> Form[TargetSpace, S]

LinearMap[MiddleSpace, TargetSpace]
    .compose(LinearMap[SourceSpace, MiddleSpace])
    -> LinearMap[SourceSpace, TargetSpace]
```

Also prove runtime source-space identity is checked once when a value enters `apply`.

**RED cases:**

- applying a map to the wrong source-degree category;
- claiming the output remains in the source degree;
- substituting different field semantics when the operator preserves semantics.

**Exit evidence:** No overload family proportional to arbitrary dimension; no type-level arithmetic simulation.

### TYPE-04 — Constrained specific-dimensional algorithm spike

**Purpose:** Prove a surface algorithm can require mathematical capability rather than a `SurfaceOneForm` class.

**Prerequisite:** `TYPE-01` through `TYPE-03`.

**Work:** Model one placeholder Hodge call requiring:

```text
TriangleManifold
WithoutBoundary
Oriented
Connected
Literal[1]
OrdinaryForm
```

Ty must accept exactly the qualified specialization and reject unknown/open/unoriented/wrong-degree/wrong-semantics inputs.

**Exit gate for T0:** If this cannot be expressed without unsafe escapes, stop and revise architecture before production code.

## T1 — Complete Simplicial Values

### CORE-01 — Array ownership and complete construction

**Purpose:** Establish the no-hack data owner pattern.

**Prerequisite:** T0 accepted.

**RED tests in `test_simplicial.py`:**

- source array mutation does not change the owner;
- exposed arrays cannot mutate retained storage under the chosen contract;
- malformed bases fail during construction;
- no zero-argument or half-initialized object exists;
- construction returns a fully printable and queryable value.

**Work:** Implement one standard `__slots__` data owner with ordinary assignments in `__init__`. Do not use frozen dataclasses for NumPy ownership.

**Review firewall:** Search for `object.__setattr__`, `__new__`, `init=False`, mutable cache fields, `Any`, casts, and ignores.

### CORE-02 — Canonical arbitrary-dimensional simplex bases

**Purpose:** Construct a valid finite complex without geometry.

**Prerequisite:** `CORE-01`.

**RED laws:**

- closure under faces;
- canonical identity independent of input row ordering policy;
- orientation kept distinct from identity;
- duplicate/degenerate/out-of-range simplices rejected;
- runtime dimension correct for arbitrary maximal-simplex width.

**Work:** Build complete simplex bases and construction-local lookup data. Do not compute or retain sparse boundary maps yet.

### CORE-03 — Boundary and incidence as intrinsic uncached methods

**Purpose:** Implement topology without instance caches.

**Prerequisite:** `CORE-02`.

**RED laws:**

- exact integer boundary coefficients;
- $B_{k-1}B_k=0$;
- unsigned incidence equals absolute boundary support;
- caller mutation of returned sparse arrays does not alter the complex or later calls.

**Work:** Expose `complex.boundary_matrix(k)` as the intrinsic API, delegate to a pure assembly helper, and return caller-owned sparse maps. No lazy dictionaries on the complex and no global `lru_cache`.

### CORE-04 — Boolean subset algebra

**Purpose:** Restore closure, star, link, purity, and boundary with one clear convention.

**Prerequisite:** `CORE-03`.

**RED laws:**

- closure is extensive, monotone, and idempotent;
- star is upward closure under the frozen definition;
- link follows one documented formula;
- purity and topological boundary are not confused with signed chain support;
- subset arrays are owned and dimension-aligned.

**Work:** Keep subset behavior within the simplicial concept. Do not create a separate topology wrapper.

### CORE-05 — Cochain spaces and forms

**Purpose:** Land the type shape proven by `TYPE-02`.

**Prerequisite:** `CORE-03`.

**RED tests:**

- a cochain space binds complex, runtime degree, basis, and coefficient count;
- forms are constructed only through a space;
- degree and semantics remain precise where statically known;
- wrong coefficient shape fails at construction;
- two runtime spaces with equal sizes remain distinct identities.

**Work:** Implement exact-space generic `Form[CochainSpace[K, Degree], Semantics]`; do not add `SurfaceOneForm`, `EdgeOneForm`, or owner tokens.

## T1 continued — Simplicial Property Refinement

### REFINE-01 — Triangle-manifold refinement

**Purpose:** Prove the capability actually consumed by specific-dimensional algorithms.

**Prerequisite:** `CORE-02` and accepted `TYPE-01` design.

**RED tests:**

- top-dimensional bases are triangles;
- edge incidence satisfies manifold policy;
- vertex links satisfy cycle/path policy;
- refinement failure leaves no stronger value;
- successful refinement preserves unrelated axes.

**Work:** Implement a constrained `triangle_manifold()` method on `Complex`; do not create `TriangleMesh` or a triangle-topology owner parallel to the complex.

### REFINE-02 — Orientation, boundary, and connectivity rules

**Purpose:** Add only independently consumed mathematical facts.

**Prerequisite:** `REFINE-01`.

**Slices:** Implement each rule as a separate subtask if the total diff approaches 500 lines.

**RED laws:**

- orientation consistency across interior faces;
- deterministic boundary classification and loops;
- connectedness from topology;
- disk-boundary evidence only when one connected boundary and Euler/topology law agree;
- each transition preserves all unrelated phantom axes.

**Work:** Keep navigation indices derived from canonical simplex bases; do not renumber edges/faces independently.

## T3 — Complete General Geometry

### GEOM-01 — Arbitrary-dimensional geometry construction and ownership

**Purpose:** Bind one exact arbitrary-dimensional complex to complete Euclidean positions and canonical simplex measures.

**Prerequisite:** `CORE-01` and `TYPE-01`.

**RED tests in `test_geometry.py`:**

- exact `float64` positions are finite, basis-aligned, owned, and complete;
- source mutation does not change geometry;
- degree-$k$ Euclidean measures agree with canonical simplex bases through arbitrary runtime dimension;
- measures scale by $s^k$ without avoidable intermediate overflow/underflow;
- valid highly anisotropic simplices are not rejected by an absolute epsilon;
- degenerate or unrepresentable geometry fails before construction returns;
- public arrays are caller-owned and no derived field is lazily stored later;
- `Geometry[K]` preserves the complete static complex specialization and exact runtime complex identity.

**Work:** Make ordinary construction and `from_positions` share one complete boundary. Normalize each local edge column, use QR diagonal products with exponent tracking, and form an exact binary-float Gram determinant from original coordinates to resolve both rank and measure when QR is suspicious. Eagerly own one measure array for every represented degree. Do not add a dimension marker, geometry state axis, metric owner, dual owner, or cache.

### GEOM-02 — Constrained triangle-geometry deductions

**Purpose:** Add only the triangle-specific fields consumed by an approved surface/DEC operation.

**Prerequisite:** `GEOM-01`, `REFINE-01`, and an accepted consumer.

**RED type/runtime tests:**

- normals/angles/cotangents are statically constrained to `TriangleManifold` geometry;
- normals additionally reject unsupported ambient dimension at runtime;
- face reorientation reverses normals but preserves measures, angles, and cotangents;
- general `Geometry[K]` remains free of optional surface-only fields.

**Work:** Prefer constrained pure methods or one consumer-owned complete computation product. Do not introduce `SurfaceGeometry`, `GeometryUnknown`, or `Nondegenerate` until a complete weaker value and real consumer prove the need.

## T4 — Typed DEC Operators

### OP-01 — Typed exterior derivative

**Purpose:** Land arbitrary-dimensional `LinearMap[SourceSpace, TargetSpace]` for topological cochain differentiation. The chain boundary remains `Complex.boundary_matrix`.

**Prerequisite:** `CORE-03`, `CORE-05`, and `TYPE-03`.

**RED laws in `test_operators.py`:**

- source/target runtime spaces agree with static categories;
- wrong runtime space is rejected at one admission boundary;
- $d_{k+1}d_k=0$ through runtime dimension eight, with durable dimension-four coverage;
- explicitly typed literals above degree two remain exact while runtime degrees remain `int`;
- operator application preserves field semantics when mathematically valid;
- composition statically unifies the intermediate degree and checks the exact runtime space;
- `matrix()` returns a complete, finite, canonicalized, caller-owned CSR representation without exposing a mutable view.

### OP-02 — Grouped primal/dual geometry measures

**Purpose:** Let one complete `Geometry` extract primal and signed circumcentric-dual measures in its canonical simplex bases.

**Prerequisite:** `GEOM-01` and `OP-01`.

**RED laws:**

- `Geometry.primal_measures(degree)` supplies $|\sigma^k|$;
- `Geometry.dual_measures(degree)` supplies signed circumcentric $|\star\sigma^k|$ in the same canonical primal basis;
- the dual cells have runtime dimension $n-k$, while both APIs consistently accept associated primal degree $k$;
- degree-two primal area comes from `Geometry.primal_measures(2)` rather than a duplicate face-area field;
- no free measure function or redundant primal/dual result wrapper exists;
- no `CircumcentricDual` object exists;
- no dual cache or duplicated `DualComplex` exists;
- zero/non-finite inverse entries are rejected only by operations requiring inversion;
- zero and non-Delaunay signs are preserved rather than silently absolutized;
- immediate-coface recurrence equals explicit signed flags through dimension four;
- degree-$k$ dual measures scale as $s^{n-k}$ and remain rigid-embedding invariant.

Two-dimensional cotangent weights are a later specialization law of the generic Hodge/Laplacian construction, not stored base-geometry data.

### OP-03 — Hodge star and weighted pairing

**Purpose:** Migrate forms/maps to exact space-generic typing, add a subordinate dual cochain space, then implement general metric DEC maps.

**Prerequisite:** `OP-02`.

**RED laws:**

- `Form[Space, Semantics]` and `LinearMap[SourceSpace, TargetSpace]` preserve exact space identity;
- the existing exterior derivative retains behavior while migrating to space-generic maps;
- `DualCochainSpace[K, PrimalDegree]` references the exact geometry and primal cochain space without duplicating topology;
- Hodge source and target spaces are explicit primal and subordinate dual spaces;
- diagonal ratio follows the frozen convention;
- weighted pairing is symmetric under its valid metric assumptions;
- returned sparse representation is caller-owned.

The exact-space migration, subordinate dual identity, forward signed Hodge map, and direct weighted pairing are implemented. Inverse Hodge remains private operation demand rather than a public API.

### OP-04 — Codifferential and Hodge Laplacian

**Purpose:** Compose typed maps rather than owner methods.

**Prerequisite:** `OP-03`.

**RED laws:**

- codifferential is the weighted adjoint;
- Laplacian maps one cochain space to itself;
- constant/nullspace laws for degree zero;
- map composition rejects incompatible spaces.

The weighted-adjoint codifferential and degree-general Hodge Laplacian are implemented, including terminal-degree logic and fail-closed zero-reciprocal handling. Signed geometry does not imply Euclidean symmetry, positive semidefiniteness, or CG suitability.

### TOPO-BC-01 — Canonical topological boundary

**Purpose:** Identify the complete topological boundary as a subset bound to the exact complex.

**Prerequisite:** `CORE-04` and the relevant manifold refinement.

**RED laws:**

- closed complexes return the empty boundary subset;
- manifold boundary faces and their closure align with canonical simplex bases;
- disconnected boundary components are preserved without renumbering;
- nonmanifold inputs fail at the owning admission boundary.

`CodimensionOneRegular` and dimension-general `topological_boundary()` are implemented. Regular admission retains immutable canonical boundary masks, and `TriangleManifold` refines that state. `without_boundary()` and `with_boundary()` classify the extracted boundary only after regular admission; zero-dimensional and boundaryless domains return canonical empty subsets, while codimension-one branching and non-pure inputs fail admission.

### OP-BC-01 — Cochain subspaces and trace maps

**Purpose:** Represent boundary/interior degrees of freedom and their exact restriction/zero-extension maps.

**Prerequisite:** `TOPO-BC-01` and `OP-01`.

**RED laws:**

- a subspace retains its parent cochain space and canonical indices;
- restriction and zero extension have the expected identity/mask laws;
- wrong-complex or wrong-degree compositions fail once at admission;
- no owner token or hidden renumbering substitutes for the actual parent space.

Strict parent-retaining `CochainSubspace`, `restrict()`, and `extend_zero()` are implemented, including empty-subspace behavior and the $RE=I$, $ER=M_I$ laws. These maps perform coefficient selection/insertion only and do not apply induced-orientation signs.

### PROBLEM-BC-01 — Explicit boundary-value assembly

**Purpose:** Apply essential and natural boundary semantics before numerical solving.

**Prerequisite:** `OP-BC-01`, the required DEC operator, and an approved boundary-value algorithm.

**RED laws:**

- Dirichlet elimination assembles $A_{II}x_I=b_I-A_{IB}g_B$ and reconstructs prescribed values exactly;
- Neumann data enters the weak-form right-hand side and incompatible forcing is rejected;
- pure-Neumann/closed nullspaces require an explicit gauge;
- Robin terms modify both operator and right-hand side;
- the numerical solver receives only an assembled system and has no boundary-condition branch.

**Work:** Keep Dirichlet, Neumann, and Robin as distinct mathematical formulations rather than a mode string, optional boundary argument, or solver configuration union.

`AssembledSystem`, endomorphism-only `eliminate_dirichlet()`, and flat `DirichletProblem.reconstruct()` are implemented. Elimination requires a `CodimensionOneRegular` primal parent, verifies the canonical topological boundary, handles empty and fully prescribed regions, and assembles $A_{II}$ and $b_I-A_{IB}g_B$ by sparse block indexing. Neumann, Robin, and compatibility/gauge assembly remain deferred.

## T5 — Numerical Behavior

### SOLVE-01 — Direct prepared solve

`prepare_direct()`, `PreparedLinearSolve`, `PrepareLinearSolve`, and flat residual-evidence `LinearSolution` are implemented. Returned direct solutions are certified against their operator and right-hand side; the complete public value constructor validates evidence consistency. `AssembledSystem.solve()` and `DirichletProblem.solve()` receive preparers explicitly; no solver strings, stored solver behavior, implicit fallback, or iterative method is present.

**Purpose:** Isolate boundary-agnostic sparse factorization and residual evidence.

**Prerequisite:** T0 type decisions.

**RED tests in `test_solvers.py`:**

- factorization prepared once for multiple RHS;
- singular and non-finite systems raise stable numerical errors;
- backend message text is not the public error contract;
- full residual is measured and certified;
- no factorization is stored on a domain value.

**Work:** Implement the minimum callable/protocol proven by the spike. Do not add a solver hierarchy or registry.

### SOLVE-02 — Iterative solve only on demonstrated demand

**Purpose:** Add CG only if an accepted study or requirement needs it.

**Prerequisite:** Direct path accepted plus explicit user approval.

**Admission:** Measured reason to avoid direct-factor storage or to compare numerical behavior.

**Work:** Bounded iterations, explicit preconditioner lifetime, convergence evidence, and no silent direct fallback.

## T6 — DEC Algorithms

### DEC-01 — Scalar Poisson problem

**Purpose:** Separate mathematical compatibility/gauge from numerical solving.

**Prerequisite:** `OP-04`, `SOLVE-01`.

**RED laws:**

- closed connected compatibility condition;
- incompatible density rejected, not silently projected;
- explicit gauge;
- scale-relative residual;
- typed degree-zero input/output;
- runtime domain identity admitted once.

### DEC-02 — Degree-one Hodge decomposition

**Purpose:** Implement the first specific-dimensional typed algorithm.

**Prerequisite:** `TYPE-04`, `OP-04`, `SOLVE-01`, and closed/oriented/connected refinements.

**RED laws:**

- only qualified triangle-manifold geometry and ordinary degree-one form typecheck;
- exact, coexact, and harmonic components reconstruct the input;
- exact/coexact orthogonality under the admitted metric;
- harmonic component is closed and coclosed;
- no unrestricted cochain path performs deep degree/surface checks.

**Output:** A complete immutable decomposition product generic over the domain, not an algorithm facade.

### DEC-03 — Homology, periods, and harmonic bases

**Purpose:** Bind topology-derived bases through real cochain spaces without owner-token glue.

**Prerequisite:** `CORE-03`, `CORE-05`, `DEC-02`.

**RED laws:**

- basis dimensions agree with topology;
- periods use the exact runtime domain once admitted;
- wrong-domain basis/form pair fails at the problem boundary;
- no repeated matching inside numerical kernels.

## T7 — Specific-Dimensional Geometry Algorithms

Each algorithm is a separate approved task and must use the exact phantom/certification requirements documented in architecture.

### SURFACE-01 — Curvature

- Require oriented complete embedded triangle-manifold geometry and the accepted triangle deductions.
- Return degree-zero forms on the same domain.
- Verify Gauss--Bonnet, scale law, and mixed-sign behavior.

### SURFACE-02 — One immutable mean-curvature-flow step

- Require complete embedded triangle geometry and positive scale-aware step.
- Readmit changed positions through complete `Geometry.from_positions`; return complete geometry or fail.
- Preserve topology state; verify centroid, dissipation, residual, and scale covariance.

### SURFACE-03 — Disk harmonic extension

- Require disk-boundary, oriented, connected triangle-manifold evidence.
- Bind boundary values to the canonical boundary space.
- Restore boundary exactly and certify residual.

### SURFACE-04 — Conformal parameterization

- Freeze the exact mathematical formulation before coding.
- Use an explicit eigen/linear solve behavior.
- Do not combine disk and closed-surface modes through a `mode` string or optional boundary argument.

### SURFACE-05 — Connection and holonomy

- Model a connection as `Form[CochainSpace[K, Literal[1]], SO2Connection]`.
- Do not permit an ordinary one-form to substitute solely because shape matches.
- Bind cycle data at one admitted problem boundary.

### SURFACE-06 — Integrability and direction fields

- Certify connection integrability as `Certified[Connection, Integrable]`.
- Direction-field integration accepts only the certified value.
- Do not discover the mathematical precondition halfway through field traversal.

## T8 — Root Public Boundary

### API-01 — Root exports

**Purpose:** Publish only implemented, verified concepts.

**Prerequisite:** At least one complete vertical slice.

**Work:** Export constructors, refinement behaviors, typed values, and the accepted algorithm from `polygeo.__init__`. Do not create `api.py` or re-export planned names.

**RED tests in the relevant existing behavior file:**

- exact public imports needed by the current slice;
- planned APIs absent;
- rejected owner-chain names absent;
- importing `polygeo` does not import Trimesh.

### API-02 — `load_surface`

**Purpose:** Keep the small mesh-loading boundary in `__init__.py`.

**Prerequisite:** Geometry construction and root API.

**Work:** Function-local Trimesh import; NPZ/mesh payload admission; return one complete geometry value with only properties actually verified by loading.

**RED tests:**

- supported formats;
- malformed payloads;
- one triangular object requirement;
- lazy Trimesh import;
- no path/I/O dependency in mathematical owners.

## T9 — Consolidated Product Verification

### VERIFY-01 — Structural firewall

Scan source and reject any match or equivalent structure:

```text
object.__setattr__
object.__dict__ mutation
custom __new__ construction gate
init=False empty shell
cached_property on domain values
owner/matches_metric identity APIs
SurfaceOneForm or combination classes
CircumcentricDual owner
Any/object/cast/type-ignore leakage in target relations
backend exception-message classification
```

AST review must supplement text search for mutable nested fields and hidden state.

### VERIFY-02 — Consolidated test ownership

Keep tests grouped by mathematical behavior:

- `test_simplicial.py`: construction, refinement, topology, spaces, forms;
- `test_geometry.py`: arbitrary-dimensional Euclidean admission, measures, scale, degeneracy, and ownership;
- `test_operators.py`: typed maps and DEC operator laws;
- `test_solvers.py`: prepared numerical behavior and residual certification;
- later solve-based DEC algorithm tests stay with their owning mathematical behavior;
- `test_surface.py`: specific-dimensional deductions and algorithms, loading, root API, installed smoke;
- `test_typing.py`: execute positive fixtures and enforce exact negative Ty rule-ID multisets in one batched process;
- `typing/*.py`: flat real-Python positive fixtures plus excluded `*_invalid.py` negative contracts.

Do not recreate one test file per implementation noun. Shared fixtures remain local until at least two behavior files need the same fixture.

### VERIFY-03 — Independent review

Review the exact diff after all gates pass. The reviewer must attack:

- phantom-state preservation;
- runtime generic erasure assumptions;
- runtime mesh/space identity admission;
- constructor completeness;
- array and sparse mutation isolation;
- surface-specific generic constraints;
- numerical signs, gauges, residuals, and scale laws;
- package imports and artifact behavior.

Any edit after review invalidates the review and requires rerunning affected gates.

### VERIFY-04 — Fresh source and artifact verification

Use a fresh isolated copy under the user's preferred temporary verification location. Verify:

1. locked dependency sync;
2. format, lint, typecheck, and full tests;
3. source consumer smoke;
4. wheel build, install, import, and representative behavior;
5. sdist build, install, import, and representative behavior;
6. root API and lazy mesh-format dependency;
7. no rejected modules or APIs in either artifact.

## Acceptance Criteria

The first architecture implementation is complete only when:

- arbitrary runtime dimension does not require one nominal type per number;
- surface-specific algorithms are constrained by verified mathematical capabilities;
- forms are generic over complex, degree, and semantics;
- no `SurfaceOneForm` or dimension-degree combination class exists;
- no explicit `TypeVar`, old-style `Generic`, `Any`, or opaque `object` appears in the approved type path;
- degree-changing operators use typed source/target spaces rather than type-level arithmetic;
- refinement transitions preserve unrelated axes and Ty proves the result;
- no hidden cache or post-construction mutation exists;
- no circumcentric-dual or form-geometry glue owner exists;
- runtime mesh/space identity is checked once at admission and not repeatedly downstream;
- all mathematical laws and negative typecheck cases pass;
- source, wheel, and sdist smokes pass in fresh environments;
- documentation describes only landed behavior.

## Stop Conditions

Stop and report before proceeding when:

- Ty cannot prove a central relationship without unsafe escapes;
- a task requires more than 500 changed lines;
- a new type exists only to forward another value;
- a property axis has no accepted algorithm consumer;
- a cache is proposed without profile evidence;
- a surface algorithm still accepts a general unrestricted form;
- runtime identity is being misrepresented as statically proven;
- a mathematical convention is unresolved;
- repository/index handling is ambiguous;
- any command would commit, push, reset, restore, or clean without explicit authorization.
