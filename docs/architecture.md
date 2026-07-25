# PolyGeo Architecture and Typed API Design

## Status

This document defines the target architecture. The arbitrary-dimensional simplicial core, constrained topology refinements, boundary matrices, simplex subsets, cochain spaces, generic forms, and complete general Euclidean geometry are implemented. Metric DEC, numerical solvers, boundary-value problems, and specific-dimensional algorithms remain target design until their implementation tasks and verification gates pass. The rejected owner-chain implementation is not part of the working API.

[`roadmap.md`](roadmap.md) contains the implementation tasks. Architecture decisions and implementation status must not be mixed.

## Goal

PolyGeo is a Python 3.14 reading implementation of discrete differential geometry and discrete exterior calculus. Its architecture should make the mathematical domain visible without turning every prerequisite into a wrapper class.

The design must support two modes honestly:

1. arbitrary-dimensional finite simplicial complexes, whose dimension and cochain degrees may be known only at runtime;
2. algorithms with stronger requirements, such as a closed oriented triangular manifold, a degree-one form, or a nondegenerate piecewise-Euclidean geometry.

The general path remains runtime-dimensional. Static specialization is introduced only after a property has been verified or when a constructor mathematically guarantees it.

## Rejected Architecture

The following chain is not part of the target design:

```text
SimplicialComplex
    -> TriangleTopology
    -> SurfaceMetric
    -> CircumcentricDual
    -> SurfaceFormGeometry
    -> EdgeOneForm
    -> HodgeDecomposition
```

It converted mathematical prerequisites into ownership checks and glue values. In particular, the target architecture forbids:

- a public `CircumcentricDual` owner;
- `SurfaceFormGeometry` or equivalent metric/dual glue;
- `_metric_owner`, `matches_metric`, or generic owner-matching APIs;
- `SurfaceOneForm`, `ClosedSurfaceOneForm`, and other dimension-degree combination classes;
- a `Result`/`unwrap` framework throughout mathematical code;
- hidden lazy mutation in frozen values;
- runtime dispatch that pretends to inspect erased generic parameters.

## Architecture Spine

```text
raw arrays / mesh payload
        |
        v
complete immutable simplicial data
        |
        +-- runtime dimension and simplex bases
        +-- verified phantom property state
        |
        v
cochain spaces and forms
        |
        +-- typed linear operators
        +-- complete piecewise-Euclidean geometry
        |
        v
DEC problem assembly
        |
        +-- injected numerical behavior
        |
        v
certified mathematical products
```

The layers are conceptual, not a demand for one source file per box.

### General simplicial algebra

Owns:

- finite oriented simplex bases;
- boundary and coboundary maps;
- Boolean subset operations;
- homological constructions;
- runtime dimension;
- cochain spaces.

It does not own positions, metric coefficients, mesh files, plotting, or numerical solvers.

### Geometry

A complete geometry composes one admitted complex with positions and all routinely required derived fields. Embedding and metric are mathematically distinct, but PolyGeo does not initially expose separate public embedding and metric owners because the current product always derives its metric from positions.

A geometry is complete when constructed. It contains no lazy metric computation and no internal operator cache.

### Metric DEC

Owns typed Hodge-star, codifferential, pairing, and Hodge-Laplacian constructions. These are not surface-specific. They consume primal/dual measures and typed cochain spaces without creating a long-lived circumcentric-dual object.

### Specific-dimensional algorithms

An algorithm is specific-dimensional when its mathematics requires a property such as triangular two-manifold topology, Euclidean embedding, closedness, or a particular cochain degree. It accepts generic values specialized by phantom properties; it does not accept an unrestricted form and then rely on documentation or repeated runtime contracts.

### Numerical behavior

Sparse factorization, iterative solving, convergence evidence, and residual measurement are replaceable behaviors. Numerical code does not know vertices, forms, curvature, genus, or boundary loops.

## Python 3.14 Type Policy

### No explicit `TypeVar`

The design may use Python 3.14 PEP 695 inline type parameters:

```python
class Complex[Boundary, Orientation, Connectivity, Topology]: ...

def transform[Domain](value: Domain) -> Domain: ...
```

It must not use old-style declarations:

```python
T = TypeVar("T")
class Value(Generic[T]): ...
```

PEP 695 parameters remain type parameters in the mathematical sense. If all type parameters were forbidden, generic input/output relationships could not be expressed.

### No opaque escape hatches

Target type surfaces must not use:

- `Any`;
- `object` as an input placeholder;
- scattered `typing.cast`;
- `# type: ignore` to force typestate transitions;
- variadic property packs whose membership Ty cannot prove.

A narrowly isolated internal cast would require an explicit design exception. The preferred refinement implementation constructs a complete typed view from immutable storage rather than casting a partially initialized value.

### Runtime generics are erased

`Complex[...]` specializations have the same runtime class. Runtime dispatch therefore uses stored marker values and behavior objects. Static generic constraints decide whether a call is type-correct; they do not perform runtime dispatch.

## Core Type Model

### Complete complex data

The array-owning runtime value is a standard `__slots__` class with read-only properties. It is not a frozen dataclass, because `frozen=True` does not make NumPy arrays immutable.

Conceptually:

```python
class Complex[
    Boundary: BoundaryState,
    Orientation: OrientationState,
    Connectivity: ConnectivityState,
    Topology: TopologyState,
]:
    @property
    def dimension(self) -> int: ...

    def connected(
        self: Complex[Boundary, Orientation, ConnectivityUnknown, Topology],
    ) -> Complex[Boundary, Orientation, Connected, Topology]: ...
```

Dimension is runtime data for the general complex. PolyGeo does not define `Dim0`, `Dim1`, ..., or one class per possible dimension.

### Why dimension is not a universal phantom axis

An arbitrary-dimensional complex may derive its dimension from input array shape. Python cannot lift an arbitrary runtime integer into `Literal[n]`, nor can it calculate `n + 1` in the type system.

Specific algorithms should require a meaningful verified capability such as `TriangleManifold`, not merely `dimension == 2`. A triangle-manifold refinement proves the two-dimensional fact together with the incidence laws the algorithm actually needs.

`Literal[2]` may still appear in a narrow type-system specialization when a fixed-dimensional constructor or algorithm benefits from it, but the general core does not enumerate dimensions.

### Phantom property axes

Common, orthogonal domain facts use fixed generic axes rather than one refinement class per property combination.

Conceptual markers include:

```text
BoundaryUnknown | Closed | HasBoundary | DiskBoundary
OrientationUnknown | Oriented
ConnectivityUnknown | Connected
TopologyUnknown | Simplicial | Manifold | TriangleManifold
```

Only properties consumed by real algorithms become axes. Complete `Geometry[K]` is a separate composition and does not add a speculative property axis to `Complex`. The architecture does not predeclare every conceivable mathematical adjective.

A single runtime `Complex` class can therefore be viewed statically as, for example:

```text
Complex[Closed, Oriented, Connected, TriangleManifold]
```

No `ClosedOrientedTriangleSurface` runtime subclass is created.

### Constrained refinement methods

A refinement method verifies one property and returns the same immutable mathematical data with a stronger phantom state. Its constrained `self` annotation fixes the source state directly, so Ty does not need to infer a generic target through a rule object.

```python
domain = (
    complex_
    .triangle_manifold()
    .oriented()
    .closed()
    .connected()
)
```

Each rule:

- verifies one mathematical property once;
- preserves unrelated generic axes;
- returns a complete value;
- does not mutate the source;
- returns a new complete instance of the same runtime `Complex` class sharing immutable internal simplicial data;
- does not introduce a refinement subclass;
- does not write a hidden cache.

T0 established that `Complex.refine(rule)` resolves its target to `Unknown` in Ty 0.0.32 and is rejected. Direct constrained methods preserve unrelated axes and reject incompatible `self` states without `Any`, casts, or ignores.

## Cochain Spaces and Forms

### Cochain space

A form is not defined by a coefficient vector and a free integer alone. It belongs to a particular ordered simplex basis.

Conceptually:

```python
class CochainSpace[
    K,
    Degree: int,
]:
    @property
    def complex(self) -> K: ...

    @property
    def degree(self) -> int: ...

    def form[Semantics: FieldSemantics](
        self,
        coefficients: FloatArray,
        semantics: Semantics,
    ) -> Form[K, Degree, Semantics]: ...
```

The space owns:

- the complex reference;
- runtime degree;
- canonical basis ordering;
- coefficient dimension.

### Generic form

```python
class Form[
    K,
    Degree: int,
    Semantics: FieldSemantics,
]:
    @property
    def space(self) -> CochainSpace[K, Degree]: ...

    @property
    def coefficients(self) -> FloatArray: ...
```

Examples are generic specializations, not subclasses:

```text
Form[K, Literal[0], OrdinaryForm]
Form[K, Literal[1], OrdinaryForm]
Form[K, Literal[2], OrdinaryForm]
Form[K, Literal[1], SO2Connection]
```

Convenience aliases may reduce annotation noise:

```python
type ZeroForm[K] = Form[K, Literal[0], OrdinaryForm]
type OneForm[K] = Form[K, Literal[1], OrdinaryForm]
type TwoForm[K] = Form[K, Literal[2], OrdinaryForm]
```

These aliases create no runtime classes and remain generic over `K`.

### General runtime degree

For a degree calculated at runtime, the honest static type uses `int`. A fixed-degree algorithm may refine or construct a `Literal[1]` space. PolyGeo does not generate one nominal type or overload family for every possible dimension.

## Typed Linear Maps

Python cannot express arbitrary type-level arithmetic such as `Degree + 1`. Operators therefore carry source and target spaces directly.

```python
class LinearMap[
    K,
    SourceDegree: int,
    TargetDegree: int,
]:
    @property
    def source(self) -> CochainSpace[K, SourceDegree]: ...

    @property
    def target(self) -> CochainSpace[K, TargetDegree]: ...

    def apply[Semantics: FieldSemantics](
        self,
        value: Form[K, SourceDegree, Semantics],
    ) -> Form[K, TargetDegree, Semantics]: ...
```

An exterior derivative is a map:

```text
CochainSpace[K, k] -> CochainSpace[K, k + 1]
```

The `k + 1` relation is checked when constructing the map at runtime. Downstream static checking compares the map's source and target degree parameters and the retained source/target space identities rather than attempting integer arithmetic.

The same model applies to:

- boundary;
- coboundary/exterior derivative;
- Hodge star;
- codifferential;
- Laplacian;
- restriction and prolongation maps.

Sparse matrices are representations of these maps, not the public mathematical identity of the operator.

## Geometry Model

### One complete general geometry value

`Geometry[K]` composes one exact simplicial complex with a finite Euclidean position matrix. Both intrinsic simplicial dimension and ambient coordinate dimension remain runtime data.

```python
class Geometry[K]:
    def __init__[
        B: BoundaryState,
        O: OrientationState,
        C: ConnectivityState,
        T: TopologyState,
    ](
        self: Geometry[Complex[B, O, C, T]],
        complex_: Complex[B, O, C, T],
        positions: FloatArray,
    ) -> None: ...

    @staticmethod
    def from_positions[
        B: BoundaryState,
        O: OrientationState,
        C: ConnectivityState,
        T: TopologyState,
    ](
        complex_: Complex[B, O, C, T],
        positions: FloatArray,
    ) -> Geometry[Complex[B, O, C, T]]: ...

    @property
    def complex(self) -> K: ...

    @property
    def ambient_dimension(self) -> int: ...

    def simplex_measures(self, degree: int) -> FloatArray: ...
```

Both ordinary construction and `from_positions` statically require an actual four-axis `Complex[...]` and use the same complete admission boundary. Construction validates and owns exact `float64` positions, requires ambient dimension at least intrinsic dimension (including valid zero-complex geometry in $\mathbb{R}^0$), and eagerly computes positive finite Euclidean measures for every canonical simplex basis. A degree-$k$ measure scales by $s^k$ under uniform position scaling. Each local edge column is normalized independently before QR factorization, and exponent-tracked products avoid avoidable intermediate overflow and underflow. Suspicious numerical rank falls back to an exact Gram determinant formed from the admitted binary-float coordinates before rounded subtraction, including scale-safe square-root conversion; no epsilon alone rejects a simplex or supplies an inaccurate near-singular measure.

Construction rejects affine degeneracy and any required public measure that is not representable. There is no speculative `GeometryUnknown`/`Nondegenerate` state axis: every returned value is complete. A future position-changing operation must call the same complete admission boundary and either return a valid `Geometry[K]` or raise `GeometryError`.

Triangle normals, corner angles, and corner cotangents are dimension-specific deductions and are not fields on the general geometry value. They require separate constrained methods or complete computation products after a consumer is approved. Primal/dual DEC measure products remain owned by the operator layer rather than duplicated on geometry.

An independently supplied intrinsic metric becomes a separate public concept only after a real caller needs intrinsic geometry without positions or multiple embeddings sharing one metric.

## Surface-Specific Algorithms Without Surface-Specific Form Classes

A surface algorithm constrains the complex and degree generically.

For closed-surface degree-one Hodge decomposition, the effective requirement is:

```text
K has TriangleManifold, Closed, Oriented, Connected
geometry is a complete piecewise-Euclidean Geometry[K]
input is Form[K, Literal[1], OrdinaryForm]
```

Conceptually:

```python
type QualifiedSurface = Complex[
    Closed,
    Oriented,
    Connected,
    TriangleManifold,
]

def hodge_decomposition(
    geometry: Geometry[QualifiedSurface],
    form: OneForm[QualifiedSurface],
    *,
    solve: LinearSolve,
) -> HodgeDecomposition[QualifiedSurface]: ...
```

The exact Python bound must be proven by the type-system spike. It must not collapse to `Any`, `object`, or a runtime-only contract.

Other requirements are similarly explicit:

| Algorithm | Required domain/form semantics |
|---|---|
| Gaussian curvature | oriented complete embedded triangle-manifold geometry; returns degree-zero form |
| Mean-curvature flow | complete embedded triangle-manifold geometry plus positive scale-aware step; output is readmitted completely or fails |
| Closed Poisson | connected closed metric triangle manifold plus compatible degree-zero density |
| Harmonic extension | connected disk-boundary triangle manifold plus boundary values in canonical boundary space |
| Hodge decomposition | connected closed oriented triangle manifold plus ordinary degree-one form |
| Holonomy | oriented closed triangle manifold plus `SO2Connection` semantics |
| Direction field | connection certified as integrable, not merely a numeric degree-one form |

No algorithm accepts booleans such as `closed=True`, a free `degree=1`, or an unrestricted cochain followed by deep validation.

## Value-Specific Certification

Properties that apply only to occasional values do not become axes on every `Form` or `Complex`.

```python
class Certified[Value, Property]:
    @property
    def value(self) -> Value: ...

    @property
    def evidence(self) -> CertificateEvidence: ...
```

Examples:

```text
Certified[Form[K, Literal[0], OrdinaryForm], MeanZero]
Certified[Form[K, Literal[1], SO2Connection], Integrable]
Certified[LinearSolution[Space], ResidualCertified]
```

This is one generic composition, not one wrapper class per property combination.

## Runtime Identity Boundary

Python cannot generate a fresh static type for every runtime complex. Two meshes may share the same static specialization while being different instances.

Therefore:

- every cochain space stores its actual complex identity;
- a form can be constructed only by its space;
- every typed operator stores its actual source and target spaces;
- multi-value admission verifies runtime space identity once;
- numerical kernels consume admitted values and do not repeat owner checks.

The architecture must not claim that a shared generic parameter alone proves runtime instance identity.

## Boundary-Value Ownership

Boundary conditions are mathematical problem assembly, not numerical solver behavior.

The ownership chain is:

```text
complex topology
  -> canonical topological-boundary subset
  -> cochain subspace plus restriction/prolongation maps
  -> formulation-specific reduced or augmented linear problem
  -> boundary-agnostic numerical solve
  -> typed full-space reconstruction and residual certificate
```

Dirichlet assembly explicitly forms

$$
A_{II}x_I = b_I - A_{IB}g_B
$$

and reconstructs the prescribed boundary values exactly. Neumann data enters the weak-form right-hand side; pure-Neumann or closed nullspaces require an explicit compatibility check and gauge. Robin data contributes to both operator and right-hand side.

Dirichlet, Neumann, and Robin formulations must not be collapsed into a mode string, optional boundary argument, solver flag, or behavior-bearing configuration union. The numerical solver receives only an admitted matrix/system and right-hand side. It never discovers boundary vertices, selects a gauge, projects incompatible forcing, or interprets cochain semantics.

Python generics cannot prove that two runtime subspaces came from one complex instance. A cochain subspace therefore retains its actual parent `CochainSpace` and canonical indices; restriction/prolongation and problem assembly verify runtime identity once before numerical kernels run. No owner token or independent boundary renumbering substitutes for that mathematical data.

## Numerical Behavior Interfaces

Replaceable behavior uses small protocols or callable aliases:

```text
PreparedLinearSolve: rhs -> solution
LinearSolve: system -> PreparedLinearSolve
EigenSolve: eigenproblem -> eigenspace
ResidualCertifier: problem x candidate -> certificate
```

No solver behavior is stored on a complex, geometry, form, or metric value. No DI container, registry, plugin manager, or behavior-bearing configuration sum type is introduced.

A solver implementation:

- catches documented backend failures;
- raises stable `NumericalError` variants;
- reports structured residual evidence;
- does not parse or expose backend exception text as domain behavior;
- may retain a factorization only inside one explicit prepared-solve lifetime.

## Error Policy

Expected invalid input and numerical failure use explicit exceptions owned by their mathematical or numerical boundary. A package-wide `Result` abstraction is not introduced.

The stable families are conceptually:

```text
SimplicialError
GeometryError
OperatorError
NumericalError
```

Programming defects, `MemoryError`, interruption, and cancellation are not converted into domain errors.

## Construction and Immutability Policy

Before designing any class, its design must answer:

1. Which computations are required for a complete valid value?
2. Which products are optional and should be external pure computations?
3. Is caching justified by measurement?

### Array-owning values

Use standard classes with:

- `__slots__`;
- complete public `__init__` or a transparent classmethod that supplies every required constructor argument;
- owned exact-dtype arrays;
- read-only properties;
- no post-construction assignment.

`dataclass(frozen=True)` is not sufficient for NumPy immutability and is not the default for array owners.

### Small immutable products

Frozen slotted dataclasses are acceptable for scalar settings and complete products whose fields are already immutable.

### Zero-tolerance structural rules

The source must contain no:

- `object.__setattr__`;
- mutation through `object.__dict__`;
- custom `__new__` used to block construction;
- `init=False` empty shells;
- post-construction field filling;
- domain-value `cached_property`;
- mutable instance cache dictionaries;
- hidden factorization retained by a domain value.

Every object is complete when construction returns.

## Compute and Cache Policy

An intrinsic computation may be exposed as a method without becoming retained data. In particular, `complex.boundary_matrix(k)` is the natural API because the canonical oriented basis of one complex uniquely determines the matrix. Its implementation delegates to a pure helper, returns a fresh caller-owned sparse array, and is not cached in T1. Metric DEC maps and expensive analyses remain external computations when they require geometry, behavior, or multiple domain values.

A global `functools.lru_cache` is not the default because NumPy-bearing values are not safely content-hashable, identity caches can retain large meshes, and cached SciPy sparse matrices can be mutated by callers.

If profiling later proves repeated construction material, the permitted order is:

1. caller retains the computed operator;
2. one explicit complete operator product is computed and caller-owned;
3. an explicit operation-scoped cache keys private immutable sparse components by immutable data identity and degree, and materializes a fresh public sparse array;
4. operation-local prepared solve retains a factorization;
5. only then consider a broader cache with documented lifetime, eviction, mutation isolation, and concurrency semantics.

No cache is stored on `Complex`, no global `lru_cache` is used, and no cache changes mathematical behavior.

## Root Public Boundary

The root package is the public composition and mesh-loading boundary. It may expose `load_surface` directly and import Trimesh only inside that function. There is no one-function `api.py` or `io.py` organizational module.

The intended public style is:

```python
from polygeo import (
    ORDINARY_FORM,
    hodge_decomposition,
    load_surface,
)

geometry = load_surface("mesh.obj")
complex_ = geometry.complex

domain = (
    complex_
    .triangle_manifold()
    .oriented()
    .closed()
    .connected()
)

space = domain.cochain_space(1)
omega = space.form(values, ORDINARY_FORM)
result = hodge_decomposition(geometry.for_complex(domain), omega, solve=direct_solve)
```

This remains a design sketch beyond the landed T1 simplicial surface. T0 proved the constrained method chain and the finite fixed-degree overloads used here.

## Review Firewall

A review immediately blocks an implementation that contains:

- `object.__setattr__`, custom `object.__new__`, or empty-shell construction;
- `Any`, opaque `object`, scattered casts, or type ignores in target type relations;
- generic property packs Ty cannot reason about;
- runtime dispatch based on `isinstance(value, Complex[...])`;
- one nominal class per dimension/property/degree combination;
- `SurfaceOneForm` or equivalent dimension-degree class;
- a long-lived circumcentric-dual or form-geometry glue owner;
- repeated owner matching inside algorithms;
- hidden mutable caches;
- backend exception-message classification;
- algorithms accepting unrestricted forms and checking dimension/degree deep inside.

Passing runtime tests does not override a structural block.

## Mathematical Verification Laws

Implementation acceptance is based on laws, not file count:

- $\partial_k\partial_{k+1}=0$;
- $d_{k+1}d_k=0$;
- typed map source and target spaces align with runtime spaces;
- refinement preserves unrelated phantom axes;
- refinement failure does not produce a stronger state;
- Hodge star follows the frozen primal/dual measure convention;
- codifferential is the weighted adjoint under the admitted metric;
- Hodge Laplacian maps one cochain space to itself;
- closed scalar Poisson enforces compatibility and gauge;
- Hodge decomposition reconstructs the input and certifies exact, coexact, and harmonic laws;
- geometry-changing operations readmit changed positions completely or fail while preserving the exact complex identity;
- source, wheel, and sdist consumers observe the same root API.

## Non-Goals

The initial architecture does not attempt:

- dependent typing of runtime mesh identity;
- type-level arbitrary integer arithmetic;
- one static type per possible dimension;
- a universal matrix backend abstraction;
- intrinsic metrics independent of positions before a caller requires them;
- global operator caching;
- compatibility with the rejected owner-chain API;
- deep package nesting or one module per mathematical noun.
