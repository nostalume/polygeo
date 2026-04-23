# PolyGeo Roadmap: Arbitrary-Dimensional Simplicial Complexes

## Current State (3D Triangle Mesh)

The existing `SimplicialComplex` is specialized for 2-manifold triangle meshes:

- Fixed dimensions: vertices (0), edges (1), faces (2)
- Precomputed `V2E` (|E|×|V|) and `E2F` (|F|×|E|)
- Hard-coded `star/closure/link/boundary` for dimensions 0/1/2
- Subset masks: `v_mask`, `e_mask`, `f_mask`

**Limitations**: cannot represent tetrahedra, 3D volumes, or higher-order simplices; operations are dimension-specific.

## Goal: General SimplicialComplex

Support complexes with simplices of **any dimension** `0 ≤ d ≤ D` (D = maximal dimension). Use **incidence matrices** as fundamental operators, enabling algebraic formulations of all topological operations.

---

## Core Design

### 1. Data Representation

**SimplicialComplex**

- `simplices: list[np.ndarray]` where `simplices[d]` is shape `(n_d, d+1)`, each row = vertex indices of a d-simplex (ordered, orientation-aware).
- `dim: int` = maximal simplex dimension present.
- `boundary_mats: dict[int, csr_matrix]` mapping `d` → `∂_d` (shape `(n_{d-1}, n_d)`).
- Lazy construction: build `∂_d` on first access via `boundary_operator(d)`.

**Why this representation?**

- Uniform across dimensions
- `∂_d` directly encodes inclusion: `∂_d[s, σ] = ±1` if σ is a face of s, sign = orientation.
- All topological operators become linear algebra on chains/cochains.

### 2. Chains and Cochains

```text
Chain space C_d  ≅  R^{n_d}   (coeffs per d-simplex)
Cochain space C^d ≅ R^{n_d}   (functions on d-simplices)

Boundary   : ∂_d : C_d → C_{d-1}    (matrix)
Coboundary : δ^{d-1} = (-1)^d ∂_d^T  (up to sign conventions)
```

- **Chains**: vectors with one entry per d-simplex (integer or real coefficients).
- **Cochains**: same shape, represent "densities" on simplices.
- All subset operations become **characteristic chains** (0/1 vectors).

### 3. Boundary Operator Construction

Given `simplices[d]` (each row = vertex indices in consistent orientation):

```python
def _build_boundary(d: int) -> csr_matrix:
    # ∂_d shape (n_{d-1}, n_d)
    # For each d-simplex s (row), list its (d) faces (d-1)-simplices.
    # Orientation sign = (-1)^i for i-th face (omit vertex i).
    # Map each face → its index in simplices[d-1] via precomputed index dict.
```

**Precomputation**: Build `(d+1)×n_d` array `faces_of_simplex[d]` for all d≥1. Also `(d)×n_{d-1}` `faces_containing_simplex[d]` (inverse) for efficient `E2F`-like builds.

**Optimization**: Precompute all boundary matrices once at construction (lazy) using adjacency dictionary maps. Complexity O(∑_d n_d·(d+1)).

### 4. Generic Subset Representation

Replace `SimplicialSubset` (3 masks) with `SimplicialSubset`:

```python
@dataclass
class SimplicialSubset:
    dim: int                     # maximal dimension of subset
    chains: dict[int, np.ndarray] # d → bool/int vector of length n_d
    parent: SimplicialComplex
```

**Semantics**: The subset is the **support** of a chain (all simplices with non-zero coefficient). For "pure" subsets, only `chains[dim]` may be non-zero after closure.

**Factory**: `complex.subset(d=2, faces=np.array([0,3]))` or `complex.subset(chains={1: np.array([...])})`.

### 5. Generic Operations (as chain maps)

All operations return `SimplicialSubset`:

#### Star

St(S) = { σ | ∃ τ∈S with σ ≤ τ }  (all faces of simplices in S)

**Algorithm**:

```python
def star(self):
    out_chains = {}
    max_d = self.dim
    for d in range(max_d, -1, -1):
        if d == max_d:
            out_chains[d] = self.chains[d].copy()
        else:
            # all d-simplices that are faces of some simplex in out_chains[d+1]
            out_chains[d] = (self.parent.boundary_operator(d+1).T @ out_chains[d+1]) != 0
    return SimplicialSubset(max_d, out_chains, self.parent)
```

**Note**: Uses `∂_{d+1}^T` (coboundary) to go downwards in dimension.

#### Closure

Cl(S) = { σ | σ ≤ τ for some τ∈S }  (all faces of simplices in S)

**Algorithm**: Same as star for a downward-closed set; but star may include cofaces. **Closure = all faces**, i.e. traversing downwards only.

```python
def closure(self):
    out_chains = {self.dim: self.chains[self.dim].copy()}
    for d in range(self.dim-1, -1, -1):
        out_chains[d] = (self.parent.boundary_operator(d+1).T @ out_chains[d+1]) != 0
    return SimplicialSubset(d, out_chains, self.parent)
```

**Difference from star**: star(S) includes any simplex having a face in S, i.e. cofaces too. For a **pure** subset, star requires traversing upwards via `∂_{d+1}` (cochain→cochain). Actually: to get cofaces, use `∂_{d+1}^T`? Wait:

- Cofaces of a d-simplex are (d+1)-simplices that have it as a face → rows of `∂_{d+1}` where column (the d-simplex) has non-zero → `∂_{d+1}` multiplied by indicator vector gives counts of how many times each (d-1)-simplex appears in boundaries; that's not cofaces.
- Cofaces: for a d-simplex s, set of (d+1)-simplices τ such that s ∈ faces(τ). This is the **transpose** of the `(d+1)`-boundary matrix: `∂_{d+1}^T` maps (d)-chains to (d+1)-chains giving "coface counts". So:
  - `∂_d`: d → d-1 (faces)
  - `∂_d^T`: d-1 → d (cofaces, i.e., simplices having a given (d-1)-simplex as a face)

Thus:

- **Closure (all faces)**: start from top-dim, repeatedly apply `∂_d^T`? No: to get *faces* we go down: from d-simplices get their (d-1)-faces via `∂_d`. So closure = repeatedly apply `∂` downwards.
- **Star (all cofaces)**: from d-simplices get their cofaces via `∂_{d+1}^T` (if d < D), and also include all faces (downwards). So star = closure ∪ (cups of faces). Actually definition: St(S) = { σ | ∃ τ∈S with σ ≤ τ } includes all σ that are faces of some τ∈S. **That is exactly the closure of S**! Wait – the standard definition: star of a simplex is the set of all simplices that contain it; star of a set is union of stars of its elements. So star(S) = { σ | ∃ τ∈S such that τ ≤ σ } (cofaces, not faces). That's *opposite*: σ is a coface of τ. So star goes **upwards** (cups). Closure goes **downwards** (faces). Many texts define link = star ∩ closure of a *single* simplex. But for a *set*, operations are:
- **Star(S)**: all simplices that have a face in S (i.e. contain some element of S) → upward closure from S.
- **Closure(S)**: all faces of simplices in S → downward closure.
- **Link(S)**: Star(S) ∩ Closure(S) (simplices disjoint from S but adjacent).

Thus:

- `Closure(S)`: start with S, go down via `∂_d` repeatedly (d, d-1, ... 0).
- `Star(S)`: start with S, go up via `∂_{d+1}^T` repeatedly (d, d+1, ... D), and *also* include all faces of those cofaces? Wait: If σ is in star, it means σ has a face in S. That does NOT require σ to be ≥ S element; it requires that some face of σ is in S. That's exactly: there exists τ ∈ S s.t. τ is a face of σ → σ is a coface of τ. So star(S) = all cofaces of S. But do we also include the faces of those cofaces? Those are already in closure(S). However, star(S) as a set may include simplices of many dimensions: all cofaces at all higher dimensions, *and also* the simplices themselves (since τ ≤ τ). That yields a union over dimensions. A simple algorithm:
  - Initialize out_chains[d] = self.chains[d] for all d.
  - For d from max_dim down to 0:
    - Propagate upward: if some d-simplex is in star, then all (d+1)-simplices having it as a face are in star. This is `out_chains[d+1] |= (∂_{d+1}^T @ out_chains[d]) > 0`.
  - Repeat until no change (or single pass top-down works if we start at lowest dim and propagate up through all levels).

**Correct star algorithm**:

```python
def star(self):
    out = {d: self.chains[d].copy() for d in self.chains}
    # propagate upwards from each dimension
    for d in range(0, self.parent.dim):
        if d in out and np.any(out[d]):
            # cofaces of d-simplices are (d+1)-simplices
            up = (self.parent.boundary_operator(d+1).T @ out[d]) > 0
            out[d+1] = out.get(d+1, False) | up
    return SimplicialSubset(out, parent)
```

**Closure algorithm**:

```python
def closure(self):
    out = {d: self.chains[d].copy() for d in self.chains}
    # propagate downwards
    for d in range(self.parent.dim, 0, -1):
        if d in out and np.any(out[d]):
            down = (self.parent.boundary_operator(d) @ out[d]) > 0  # faces
            out[d-1] = out.get(d-1, False) | down
    return SimplicialSubset(out, parent)
```

**Link**: `Lk(S) = St(S) ∩ Cl(S)` interpreted as simplices disjoint from S? Standard definition: link of a set S is { σ | σ ∩ S = ∅ and σ ∪ S is a simplex? Actually for a *single* simplex σ, link = star(σ) ∩ closure(σ). For a set, often defined similarly. SimplicialSubset's link should return `star() & closure()` elementwise? But that intersection would be simplices that are both cofaces and faces of S; that's the "middle" layer. Indeed link(S) = St(S) ∩ Cl(S). This yields simplices disjoint from S but adjacent.

#### IsComplex

S is a simplicial complex iff `S == closure(S)` (downward closed). Equivalent to: no d-simplex present without all its faces.

```python
def is_complex(self):
    cl = self.closure()
    return all(np.array_equal(self.chains.get(d, False), cl.chains[d]) for d in cl.chains)
```

#### IsPure

S is pure of degree k if:

- All maximal simplices have same dimension k.
- Every simplex of dimension < k is face of some k-simplex in S.

**Algorithm**:

1. Compute `max_d = max { d | chains[d] nonempty }`.
2. For each `d < max_d`: every d-simplex in S must be face of some (d+1)-simplex in S. Use `∂_{d+1}` incidence: `(∂_{d+1} @ chains[d+1]) > 0` yields d-simplices that are faces of some (d+1)-simplex. Must cover `chains[d]`.
3. Also check no simplices of dimension > max_d (obvious).

Returns `(is_pure, max_d or -1)`.

#### Boundary

For a pure k-complex, boundary consists of (k-1)-simplices incident to **exactly one** k-simplex in S. Compute via `∂_k` restricted to S:

```python
def boundary(self):
    is_pur, k = self.is_pure()
    if not is_pur or k == 0: raise or return empty
    bd_counts = self.parent.boundary_operator(k) @ self.chains[k].astype(float)
    bd_mask = bd_counts == 1
    # collect lower dims via faces of boundary simplices? Actually boundary ∂S is a (k-1)-chain; to return full subset we need all faces of those boundary (k-1)-simplices? The boundary *as a subset* is the support of the (k-1)-chain: all (k-1)-simplices with coefficient ≡1 mod 2? Typically boundary of a chain is a chain; as a subset, we include those (k-1)-simplices. So output subset has only dimension (k-1) non-zero.
    return SimplicialSubset({k-1: bd_mask}, parent)
```

But our earlier boundary returned all faces of boundary edges etc. Actually for a pure k-complex, `∂(S)` is a (k-1)-chain. Its **support** is the set of (k-1)-simplices that appear. That's the boundary subset. So we return subset of dimension `k-1` only.

---

## Implementation Phases

### Phase 1: Generic Incidence Machinery

- Build `simplices` list from vertex list and top-level simplex arrays (user provides: list of arrays per dimension, or a single array + dimension parameter).
- Implement `_build_boundary(d)` that constructs `∂_d` as sparse matrix using `simplices[d]` and `simplices[d-1]`. Precompute index maps: `simplex_index[d][tuple(sorted(vs))] → index`.
- Provide `boundary_operator(d)` property (lazy, cached).
- Unit test: check `∂_1 ∂_2 = 0` (boundary of boundaryzero) on small complexes.

### Phase 2: Generic SimplicialSubset

- Represent as `chains: dict[int, np.ndarray]` (bool or int).
- `subset(d, indices=None, mask=None)` factory: constructs a pure-d subset or mixed if multiple dims given.
- `is_complex()`: compare with closure.
- `closure()`: downward via `∂`.
- `star()`: upward via `∂^T` plus downward? Wait: star = all simplices having a face in S. That equals: start with S; for each dimension d from min(S) to D-1, compute cofaces of current d-simplices via `∂_{d+1}^T` and add to (d+1); then also add *all faces* of those cofaces? Actually if σ is a coface of some τ∈S, then σ is in star. The faces of σ are not necessarily in star unless they are also cofaces of something in S (they might be, since if σ has a face in S, some face of σ might also have a face in S? Not guaranteed). Standard star of a *vertex* includes all simplices that contain that vertex; that includes edges, faces, tetrahedra etc. It does *not* include faces of those faces that don't contain the vertex. So star = all simplices σ such that ∃ τ∈S with τ ⊆ σ. That's upward closure only, not including faces of cofaces that don't contain S. So algorithm: start with S; for d from min_d to D-1, compute `out[d+1] |= ∂_{d+1}^T @ out[d] > 0`. No downward propagation. **Closure** does downward. So for a vertex v, star(v) = {σ | v ∈ σ} = all simplices incident to v; that's cofaces only.
- `link()`: `star() & closure()` intersection? Actually link(S) = star(S) ∩ closure(S) for a set S (sometimes also requires disjointness? For a single simplex, link = star ∩ closure with that simplex removed). For a set S, link = { σ | σ ∩ S = ∅ and σ ∪ S is a simplex } but easier: `link(S) = closure(star(S)) ∩ star(closure(S))`? Simpler: `link(S) = star(S) ∩ closure(S)` and then remove S? In many definitions, link of a face is St(F) ∩ Cl(F) with F removed. For our subset S, we can compute `lk = star() & closure()`, then subtract S if needed. But earlier version returned `star & ~closure` (set difference). That gave edges/faces *adjacent* but not containing. Correction: For vertex v, star(v) includes faces containing v; closure(v) = {v}. Link(v) = star(v) ∖ closure(v) = edges/faces containing v, with v removed. That matches earlier: link had only edges/faces, no vertices. That's the correct geometric link. So `link = star() - closure()`.
- `is_pure()`: check closure equals union of all maximal simplices? Implementation: find max d with non-zero; verify every lower-d simplex in subset is face of some higher-d simplex in subset (using `∂`). Also verify subset is a complex first (or incorporate).
- `boundary()`: for pure k, compute `∂_k @ chains[k]`. Support of that (k-1)-chain is boundary.

### Phase 3: API and Usability

- Provide convenience methods: `complex.star(d, indices)`, `complex.closure(d, indices)`.
- Support mixed-dimension subsets via combined masks.
- `complex.pure_subset(d, selection)` returns pure d-complex (closure of selection).
- `complex.boundary_of_subset(...)`.

### Phase 4: Integration and Performance

- Lazy evaluation + caching of boundary matrices.
- Use `scipy.sparse.csr` for fast matrix-vector multiplies.
- For large complexes, consider `np.bool_` accumulators or sparse boolean algebra.
- Benchmark on tetrahedral mesh (3D volumes) vs. surface mesh.

### Phase 5: Applications Layer

- Laplace–Beltrami on functions (0-cochains) via `∂^* ∂` (away from boundary).
- Hodge Laplacian on `k`-forms: `Δ = d δ + δ d`.
- Solver integration (conjugate gradient).
- Export results to polyscope/trimesh.

---

## API Sketch

```python
# Construction from tetrahedral mesh
V = ...  # (n_v, 3)
T = ...  # (n_tet, 4) vertex indices
sc = SimplicialComplex.from_tetrahedra(V, T)

# Or from generic list
sc = SimplicialComplex(simplices=[
    np.arange(n_v).reshape(-1,1),          # 0-simplices (vertices)
    edges,                                 # (n_e,2)
    faces,                                # (n_f,3)
    tetrahedra                           # (n_t,4) optional
])

# Subsets
v0 = sc.subset(dim=0, indices=np.array([0]))   # single vertex
star_v0 = v0.star()    # all incident edges, faces, tetrahedra...

# Pure 2-complex from faces
surf = sc.subset(dim=2, faces=np.array([0,2,5]))
is_pure, deg = surf.is_pure()  # (True,2)

# Boundary of a 2-complex (surface)
bd_surf = surf.boundary()  # returns 1-chain (edges)

# Chain operations (advanced)
C2 = sc.chain_space(dim=2)   # shape (n_f,)
d1 = sc.boundary_operator(1)  # ∂_1: E→V
cofaces = d1.T @ some_edge_mask
```

---

## Testing Strategy

- Unit tests for `∂_d ∂_{d+1} = 0` (exactness).
- `closure()` idempotent, `star()` monotone, `link()` definition.
- `is_complex` for all subsets of small complexes.
- Boundary double-application = 0 on pure complexes.
- Tetrahedral cube, 3D torus, higher-dim analogues.

---

## Risks & Mitigations

- **Memory**: boundary matrices can be large for dense meshes. Use sparse CSR; build on demand per dimension.
- **Orientation signs**: need consistent orientation ordering. We adopt `(-1)^i` for i-th opposite face. Verify `∂^2 = 0` in tests.
- **API complexity**: hide matrices behind high-level subset methods. Provide both chain-level and subset-level interfaces.

---

## Milestones

1. `M1`: Generic boundary matrix builder; `∂_2 ∂_1 = 0` passes on tetrahedron.
2. `M2`: `SimplicialSubset` with `closure()`, `star()` for any dimension.
3. `M3`: `is_complex`, `is_pure`, `boundary` correctly on mixed dims.
4. `M4`: Integration test suite (covers previous 3D tests as special case).
5. `M5`: DEC operators (codifferential, Laplacian) on 0/1/2-forms.
