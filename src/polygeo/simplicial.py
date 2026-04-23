"""Generic simplicial complex operations for arbitrary dimensions.

Provides:
- SimplicialComplex: multi-dimensional simplicial complexes with lazy boundary operators.
- SimplicialSubset: subsets represented as chains (dim -> bool/int vectors).

Key concepts:
  Chain spaces C_d ≅ R^{n_d}, basis = d-simplices.
  Boundary operator ∂_d : C_d → C_{d-1} (signed integer matrix).
  Unsigned incidence I_d = |∂_d| (non-zero entries = 1) for set-theoretic operations.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class SimplicialSubset:
    """A subset of a simplicial complex, represented as chains over dimensions.

    Chains: dict mapping dimension -> bool/int array of length n_d.
    Parent: reference to the owning SimplicialComplex.
    """

    chains: dict[int, np.ndarray]
    parent: "SimplicialComplex" = field(repr=False)

    def __post_init__(self):
        """Validate chains: each dimension must be within parent and have correct length."""
        for d, arr in self.chains.items():
            assert d <= self.parent.dim, (
                f"Dimension {d} exceeds complex dim {self.parent.dim}"
            )
            expected = self.parent.n_simplices(d)
            assert arr.shape[0] == expected, (
                f"Chain dim {d} length mismatch: {arr.shape[0]} vs {expected}"
            )

    @property
    def dim(self) -> int:
        """Return maximal dimension with non-zero entries, or -1 if empty."""
        if not self.chains:
            return -1
        present = [d for d, arr in self.chains.items() if np.any(arr)]
        return max(present) if present else -1

    def is_complex(self) -> bool:
        """Check if subset is a simplicial complex (downward-closed).

        A subset is a complex iff it equals its closure.
        """
        cl = self.closure()
        for d in range(self.parent.dim + 1):
            a = self.chains.get(d, np.zeros(self.parent.n_simplices(d), dtype=bool))
            b = cl.chains.get(d, np.zeros(self.parent.n_simplices(d), dtype=bool))
            if not np.array_equal(a, b):
                return False
        return True

    def closure(self) -> "SimplicialSubset":
        """Return downward closure of the subset.

        Starting from current chains, repeatedly apply boundary operator
        to propagate selections downward to all faces.
        """
        out = {d: arr.copy() for d, arr in self.chains.items()}
        for d in range(self.parent.dim, 0, -1):
            if d in out and np.any(out[d]):
                # Get faces of selected d-simplices via incidence
                faces = (self.parent.incidence_operator(d) @ out[d].astype(float)) > 0
                prev = out.get(
                    d - 1, np.zeros(self.parent.n_simplices(d - 1), dtype=bool)
                )
                out[d - 1] = prev | faces
        # Ensure all dimensions up to parent.dim are present
        for d in range(0, self.parent.dim + 1):
            if d not in out:
                out[d] = np.zeros(self.parent.n_simplices(d), dtype=bool)
        return SimplicialSubset(out, self.parent)

    def star(self) -> "SimplicialSubset":
        """Return upward star of the subset.

        Starting from current chains, propagate upward to all simplices
        that contain any selected simplex as a face (via transpose of incidence).
        """
        out = {d: arr.copy() for d, arr in self.chains.items()}
        for d in range(0, self.parent.dim):
            if d in out and np.any(out[d]):
                # Get cofaces (simplices having this as face)
                cofaces = (
                    self.parent.incidence_operator(d + 1).T @ out[d].astype(float)
                ) != 0
                prev = out.get(
                    d + 1, np.zeros(self.parent.n_simplices(d + 1), dtype=bool)
                )
                out[d + 1] = prev | cofaces
        # Ensure all dimensions up to parent.dim are present
        for d in range(0, self.parent.dim + 1):
            if d not in out:
                out[d] = np.zeros(self.parent.n_simplices(d), dtype=bool)
        return SimplicialSubset(out, self.parent)

    def link(self) -> "SimplicialSubset":
        """Return link Cl(St(S)) \ St(Cl(S))

        Link contains simplices that are in the star but not in the closure,
        i.e., adjacent simplices disjoint from the subset.
        """
        st = self.star()
        cl = self.closure()
        dims = set(st.chains) | set(cl.chains)
        out = {}
        for d in dims:
            a = st.chains.get(d, np.zeros(self.parent.n_simplices(d), dtype=bool))
            b = cl.chains.get(d, np.zeros(self.parent.n_simplices(d), dtype=bool))
            out[d] = a & ~b
        return SimplicialSubset(out, self.parent)

    def is_pure(self) -> tuple[bool, int]:
        """Check if subset is a pure complex of degree k.

        Pure complex: all maximal simplices have same dimension k,
        and every lower simplex is face of some k-simplex in the subset.
        Returns (is_pure, k) or (True, -1) if empty.
        """
        if not self.is_complex():
            return False, -1
        max_d = -1
        for d in range(self.parent.dim, -1, -1):
            if d in self.chains and np.any(self.chains[d]):
                max_d = d
                break
        if max_d == -1:
            return True, -1
        for d in range(0, max_d):
            present = self.chains.get(
                d, np.zeros(self.parent.n_simplices(d), dtype=bool)
            )
            if not np.any(present):
                continue
            higher = self.chains.get(
                d + 1, np.zeros(self.parent.n_simplices(d + 1), dtype=bool)
            )
            if not np.any(higher):
                return False, -1
            # Check all present d-simplices are covered by some (d+1)-simplex
            covered = (self.parent.incidence_operator(d + 1) @ higher.astype(float)) > 0
            if not np.all(present <= covered):
                return False, -1
        return True, max_d

    def boundary(self) -> "SimplicialSubset":
        """Return boundary of a pure k-complex (k-1 simplices with odd incidence).

        For a pure k-complex, boundary consists of (k-1)-simplices
        that are incident to exactly one k-simplex in the subset.
        """
        pure, k = self.is_pure()
        if not pure:
            raise ValueError("Subset must be pure to compute boundary")
        if k <= 0:
            return SimplicialSubset({}, self.parent)
        k_chain = self.chains.get(k, np.zeros(self.parent.n_simplices(k), dtype=bool))
        bd_counts = self.parent.boundary_operator(k) @ k_chain.astype(float)
        # Boundary simplices have odd count (typically 1 for manifold)
        bd_mask = np.abs(bd_counts) == 1
        bd_subset = SimplicialSubset({k - 1: bd_mask.astype(bool)}, self.parent)
        return bd_subset.closure()


class SimplicialComplex:
    """Arbitrary-dimensional simplicial complex with lazy boundary/incidence matrices.

    Data:
      V: (n_0, D) vertex coordinates (D = ambient dimension, e.g. 2 or 3).
      simplices[d]: (n_d, d+1) array of vertex indices (d >= 0).
      dim: maximal simplex dimension present.
    Boundary matrices ∂_d are built lazily and cached.
    """

    def __init__(self, V: np.ndarray, simplices: list[np.ndarray]):
        """Initialize from vertex coordinates and simplex arrays per dimension.

        V: (n_0, D) vertex coordinates.
        simplices[0]: (n_0, 1) — usually arange, but kept for uniformity.
        simplices[d] for d >= 1: (n_d, d+1) vertex-index arrays.
        """
        assert len(simplices) > 0, "Complex must have at least vertices"
        assert V.ndim == 2, "V must be a 2-D array of coordinates"
        self.V: np.ndarray = V.astype(float)
        self.simplices = [arr.astype(int) for arr in simplices]
        self.dim: int = len(simplices) - 1
        n_verts = V.shape[0]
        # Validate: simplices[0] should index into V
        assert self.simplices[0].shape[0] == n_verts
        for d in range(1, self.dim + 1):
            assert self.simplices[d].ndim == 2 and self.simplices[d].shape[1] == d + 1
            assert np.all(self.simplices[d] >= 0) and np.all(
                self.simplices[d] < n_verts
            )
        self._boundary: dict[int, csr_matrix] = {}
        self._incidence: dict[int, csr_matrix] = {}

    @classmethod
    def from_mesh_file(cls, mesh_path: str) -> "SimplicialComplex":
        """Build from a mesh file using gpytoolbox."""
        import gpytoolbox as gpy

        result = gpy.read_mesh(mesh_path)
        V, F = result[0:2]
        assert V is not None and F is not None, "Failed to read mesh"
        return cls.from_mesh(V, F)

    @classmethod
    def from_mesh(cls, V: np.ndarray, F: np.ndarray) -> "SimplicialComplex":
        """Build 2D simplicial complex from vertex coordinates V and triangle face array F."""
        edges_set = set()
        for tri in F:
            for i in range(3):
                edges_set.add(tuple(sorted([tri[i], tri[(i + 1) % 3]])))
        E = np.array(sorted(edges_set), dtype=int)
        V_arr = np.arange(V.shape[0]).reshape(-1, 1)
        return cls(V, [V_arr, E, F.astype(int)])

    def n_simplices(self, d: int) -> int:
        """Return number of d-simplices."""
        return self.simplices[d].shape[0]

    def _simplex_index(self, d: int) -> dict[tuple, int]:
        """Build {sorted_vertex_tuple: index} for d-simplices."""
        idx = {}
        arr = self.simplices[d]
        for i, row in enumerate(arr):
            idx[tuple(sorted(row))] = i
        return idx

    def boundary_operator(self, d: int) -> csr_matrix:
        """Return ∂_d: C_d → C_{d-1} (cached). Shape (n_{d-1}, n_d)."""
        if d not in self._boundary:
            if d == 0:
                self._boundary[0] = csr_matrix((0, self.n_simplices(0)))
            else:
                faces = self.simplices[d]  # (n_d, d+1)
                idx_map = self._simplex_index(d - 1)
                rows, cols, vals = [], [], []
                for j, simplex in enumerate(faces):
                    for i in range(d + 1):
                        face_verts = tuple(sorted(np.delete(simplex, i)))
                        row = idx_map[face_verts]
                        sign = (-1) ** i
                        rows.append(row)
                        cols.append(j)
                        vals.append(sign)
                self._boundary[d] = csr_matrix(
                    (vals, (rows, cols)),
                    shape=(self.n_simplices(d - 1), self.n_simplices(d)),
                )
        return self._boundary[d]

    def incidence_operator(self, d: int) -> csr_matrix:
        """Return |∂_d| (unsigned incidence, cached). Shape (n_{d-1}, n_d)."""
        if d not in self._incidence:
            bd = self.boundary_operator(d)
            self._incidence[d] = csr_matrix(
                (np.abs(bd.data), bd.indices, bd.indptr), shape=bd.shape
            )
        return self._incidence[d]

    def __repr__(self):
        parts = [f"d={d}:{self.n_simplices(d)}" for d in range(self.dim + 1)]
        return f"SimplicialComplex({', '.join(parts)}, V={self.V.shape})"
