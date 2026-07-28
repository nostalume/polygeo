"""Finite simplicial complexes, typestate refinements, and cochains."""

from __future__ import annotations

from itertools import combinations
from typing import Literal, Protocol, Self, overload

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csgraph, csr_array

type IndexArray = NDArray[np.int64]
type SignArray = NDArray[np.int8]
type BoolArray = NDArray[np.bool_]
type CoefficientArray = NDArray[np.float64]


class SimplicialError(ValueError):
    """Invalid simplicial input, state refinement, or cochain admission."""


class BoundaryState:
    __slots__ = ()


class BoundaryUnknown(BoundaryState):
    __slots__ = ()


class WithoutBoundary(BoundaryState):
    __slots__ = ()


class WithBoundary(BoundaryState):
    __slots__ = ()


class OrientationState:
    __slots__ = ()


class OrientationUnknown(OrientationState):
    __slots__ = ()


class Oriented(OrientationState):
    __slots__ = ()


class ConnectivityState:
    __slots__ = ()


class ConnectivityUnknown(ConnectivityState):
    __slots__ = ()


class Connected(ConnectivityState):
    __slots__ = ()


class TopologyState:
    __slots__ = ()


class Simplicial(TopologyState):
    __slots__ = ()


class CodimensionOneRegular(TopologyState):
    """A pure complex with one or two top cofaces per codimension-one simplex."""

    __slots__ = ()


class TriangleManifold(CodimensionOneRegular):
    __slots__ = ()


class FieldSemantics:
    __slots__ = ()


class OrdinaryForm(FieldSemantics):
    __slots__ = ()


ORDINARY_FORM = OrdinaryForm()


def _owned_array[DType: np.generic](
    value: NDArray[DType], dtype: np.dtype[DType]
) -> NDArray[DType]:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _permutation_sign(row: IndexArray) -> np.int8:
    inversions = 0
    for left in range(len(row)):
        for right in range(left + 1, len(row)):
            inversions += int(row[left] > row[right])
    return np.int8(-1 if inversions % 2 else 1)


class _ComplexData:
    __slots__ = ("_vertex_count", "_dimension", "_simplices", "_orientations")

    def __init__(
        self,
        *,
        vertex_count: int,
        dimension: int,
        simplices: tuple[IndexArray, ...],
        orientations: tuple[SignArray, ...],
    ) -> None:
        if vertex_count < 1 or dimension < 0:
            raise SimplicialError("a complex needs vertices and nonnegative dimension")
        if len(simplices) != dimension + 1 or len(orientations) != len(simplices):
            raise SimplicialError("simplicial bases must cover every degree")
        owned_simplices: list[IndexArray] = []
        owned_orientations: list[SignArray] = []
        for degree, (basis, signs) in enumerate(
            zip(simplices, orientations, strict=True)
        ):
            if basis.ndim != 2 or basis.shape[1] != degree + 1:
                raise SimplicialError("simplex basis has an invalid shape")
            if signs.shape != (len(basis),):
                raise SimplicialError("orientation signs do not align with the basis")
            owned_simplices.append(_owned_array(basis, np.dtype(np.int64)))
            owned_orientations.append(_owned_array(signs, np.dtype(np.int8)))
        self._vertex_count = vertex_count
        self._dimension = dimension
        self._simplices = tuple(owned_simplices)
        self._orientations = tuple(owned_orientations)


_EVIDENCE_TOKEN = object()


class _BoundaryEvidence:
    __slots__ = ("_data", "_masks", "_sealed")
    _data: _ComplexData
    _masks: tuple[BoolArray, ...]
    _sealed: bool

    def __init__(
        self,
        token: object,
        data: _ComplexData,
        masks: tuple[BoolArray, ...],
    ) -> None:
        if token is not _EVIDENCE_TOKEN:
            raise SimplicialError("boundary evidence is package-private")
        if len(masks) != data._dimension + 1:
            raise SimplicialError("boundary evidence must cover every degree")
        owned: list[BoolArray] = []
        for degree, mask in enumerate(masks):
            candidate = np.asarray(mask)
            if candidate.dtype.kind != "b" or candidate.shape != (
                len(data._simplices[degree]),
            ):
                raise SimplicialError(
                    "boundary evidence does not align with its complex"
                )
            owned.append(_owned_array(candidate, np.dtype(np.bool_)))
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_masks", tuple(owned))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("boundary evidence is immutable")
        object.__setattr__(self, name, value)


class Complex[
    B: BoundaryState,
    O: OrientationState,
    C: ConnectivityState,
    T: TopologyState,
]:
    """A complete finite simplicial complex with verified phantom state."""

    __slots__ = (
        "_data",
        "_boundary_state",
        "_orientation_state",
        "_connectivity_state",
        "_topology_state",
        "_boundary_evidence",
        "_sealed",
    )
    _data: _ComplexData
    _boundary_state: B
    _orientation_state: O
    _connectivity_state: C
    _topology_state: T
    _boundary_evidence: _BoundaryEvidence | None
    _sealed: bool

    def __init__(
        self,
        data: _ComplexData,
        boundary_state: B,
        orientation_state: O,
        connectivity_state: C,
        topology_state: T,
        boundary_evidence: _BoundaryEvidence | None = None,
    ) -> None:
        authentic_evidence = (
            type(boundary_evidence) is _BoundaryEvidence
            and boundary_evidence._data is data
        )
        if isinstance(topology_state, CodimensionOneRegular):
            if not authentic_evidence:
                raise SimplicialError(
                    "codimension-one regular state requires verified topology evidence"
                )
        elif boundary_evidence is not None:
            raise SimplicialError(
                "simplicial topology cannot carry codimension-one boundary evidence"
            )

        classified = isinstance(boundary_state, (WithoutBoundary, WithBoundary))
        if classified:
            if not authentic_evidence:
                raise SimplicialError(
                    "classified boundary state requires regular topology evidence"
                )
            if type(boundary_evidence) is not _BoundaryEvidence:
                raise SimplicialError(
                    "classified boundary state requires regular topology evidence"
                )
            actual = data._dimension > 0 and bool(
                boundary_evidence._masks[data._dimension - 1].any()
            )
            expected = isinstance(boundary_state, WithBoundary)
            if actual != expected:
                raise SimplicialError(
                    "classified boundary state conflicts with topology evidence"
                )
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_boundary_state", boundary_state)
        object.__setattr__(self, "_orientation_state", orientation_state)
        object.__setattr__(self, "_connectivity_state", connectivity_state)
        object.__setattr__(self, "_topology_state", topology_state)
        object.__setattr__(self, "_boundary_evidence", boundary_evidence)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("complexes are immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def from_maximal_simplices(
        _cls,
        maximal_simplices: IndexArray,
        *,
        vertex_count: int | None = None,
    ) -> Complex[
        BoundaryUnknown,
        OrientationUnknown,
        ConnectivityUnknown,
        Simplicial,
    ]:
        source = np.asarray(maximal_simplices)
        if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
            raise SimplicialError("maximal simplices must be a nonempty matrix")
        if source.dtype.kind not in "iu" or source.dtype.kind == "b":
            raise SimplicialError("vertex indices must be integers")
        rows = np.array(source, dtype=np.int64, order="C", copy=True)
        if np.any(rows < 0):
            raise SimplicialError("vertex indices must be nonnegative")
        for row in rows:
            if len(np.unique(row)) != len(row):
                raise SimplicialError("a simplex cannot repeat a vertex")

        canonical = np.sort(rows, axis=1)
        identities = [tuple(int(v) for v in row) for row in canonical]
        if len(set(identities)) != len(identities):
            raise SimplicialError("duplicate maximal simplex identity")

        maximum = int(rows.max())
        admitted_vertices = maximum + 1 if vertex_count is None else vertex_count
        if admitted_vertices <= maximum or admitted_vertices < 1:
            raise SimplicialError("vertex_count does not contain every index")

        dimension = rows.shape[1] - 1
        bases: list[IndexArray] = []
        for degree in range(dimension + 1):
            if degree == 0:
                keys = [(vertex,) for vertex in range(admitted_vertices)]
            else:
                keys = sorted(
                    {
                        face
                        for maximal in identities
                        for face in combinations(maximal, degree + 1)
                    }
                )
            bases.append(np.array(keys, dtype=np.int64).reshape(-1, degree + 1))

        top_sign_by_identity = {
            identity: _permutation_sign(row)
            for identity, row in zip(identities, rows, strict=True)
        }
        orientations: list[SignArray] = [
            np.ones(len(basis), dtype=np.int8) for basis in bases
        ]
        orientations[dimension] = np.array(
            [
                top_sign_by_identity[tuple(int(v) for v in row)]
                for row in bases[dimension]
            ],
            dtype=np.int8,
        )
        data = _ComplexData(
            vertex_count=admitted_vertices,
            dimension=dimension,
            simplices=tuple(bases),
            orientations=tuple(orientations),
        )
        return Complex(
            data,
            BoundaryUnknown(),
            OrientationUnknown(),
            ConnectivityUnknown(),
            Simplicial(),
        )

    @property
    def vertex_count(self) -> int:
        return self._data._vertex_count

    @property
    def dimension(self) -> int:
        return self._data._dimension

    @property
    def boundary_state(self) -> B:
        return self._boundary_state

    @property
    def orientation_state(self) -> O:
        return self._orientation_state

    @property
    def connectivity_state(self) -> C:
        return self._connectivity_state

    @property
    def topology_state(self) -> T:
        return self._topology_state

    def simplex_count(self, degree: int) -> int:
        self._validate_degree(degree)
        return len(self._data._simplices[degree])

    def simplices(self, degree: int) -> IndexArray:
        self._validate_degree(degree)
        return self._data._simplices[degree].copy()

    def orientations(self, degree: int) -> SignArray:
        self._validate_degree(degree)
        return self._data._orientations[degree].copy()

    def shares_data_with[
        B2: BoundaryState,
        O2: OrientationState,
        C2: ConnectivityState,
        T2: TopologyState,
    ](self, other: Complex[B2, O2, C2, T2]) -> bool:
        return self._data is other._data

    def boundary_matrix(self, degree: int) -> csr_array:
        if degree < 0 or degree > self.dimension:
            raise SimplicialError("boundary degree is outside the complex")
        return _assemble_boundary_matrix(self._data, degree)

    def subset(
        self, masks: tuple[BoolArray, ...]
    ) -> SimplexSubset[Complex[B, O, C, T]]:
        return SimplexSubset(self, masks)

    def triangle_manifold(
        self: Complex[B, O, C, Simplicial],
    ) -> Complex[B, O, C, TriangleManifold]:
        boundary_masks = _require_triangle_manifold(self)
        boundary_evidence = _BoundaryEvidence(
            _EVIDENCE_TOKEN, self._data, boundary_masks
        )
        return Complex(
            self._data,
            self.boundary_state,
            self.orientation_state,
            self.connectivity_state,
            TriangleManifold(),
            boundary_evidence,
        )

    def codimension_one_regular(
        self: Complex[B, O, C, Simplicial],
    ) -> Complex[B, O, C, CodimensionOneRegular]:
        boundary_masks = _require_codimension_one_regular(self)
        boundary_evidence = _BoundaryEvidence(
            _EVIDENCE_TOKEN, self._data, boundary_masks
        )
        return Complex(
            self._data,
            self.boundary_state,
            self.orientation_state,
            self.connectivity_state,
            CodimensionOneRegular(),
            boundary_evidence,
        )

    def oriented(
        self: Complex[B, OrientationUnknown, C, T],
    ) -> Complex[B, Oriented, C, T]:
        _require_oriented(self)
        return Complex(
            self._data,
            self.boundary_state,
            Oriented(),
            self.connectivity_state,
            self.topology_state,
            self._boundary_evidence,
        )

    def without_boundary[T2: CodimensionOneRegular](
        self: Complex[BoundaryUnknown, O, C, T2],
    ) -> Complex[WithoutBoundary, O, C, T2]:
        _require_boundary_extent(self, present=False)
        return Complex(
            self._data,
            WithoutBoundary(),
            self.orientation_state,
            self.connectivity_state,
            self.topology_state,
            self._boundary_evidence,
        )

    def with_boundary[T2: CodimensionOneRegular](
        self: Complex[BoundaryUnknown, O, C, T2],
    ) -> Complex[WithBoundary, O, C, T2]:
        _require_boundary_extent(self, present=True)
        return Complex(
            self._data,
            WithBoundary(),
            self.orientation_state,
            self.connectivity_state,
            self.topology_state,
            self._boundary_evidence,
        )

    def connected(
        self: Complex[B, O, ConnectivityUnknown, T],
    ) -> Complex[B, O, Connected, T]:
        _require_connected(self)
        return Complex(
            self._data,
            self.boundary_state,
            self.orientation_state,
            Connected(),
            self.topology_state,
            self._boundary_evidence,
        )

    @overload
    def cochain_space(self, degree: Literal[0]) -> CochainSpace[Self, Literal[0]]: ...

    @overload
    def cochain_space(self, degree: Literal[1]) -> CochainSpace[Self, Literal[1]]: ...

    @overload
    def cochain_space(self, degree: Literal[2]) -> CochainSpace[Self, Literal[2]]: ...

    @overload
    def cochain_space[Degree: int](
        self, degree: Degree
    ) -> CochainSpace[Self, Degree]: ...

    def cochain_space[Degree: int](self, degree: Degree) -> CochainSpace[Self, Degree]:
        self._validate_degree(degree)
        return CochainSpace(self, degree)

    def _validate_degree(self, degree: int) -> None:
        if degree < 0 or degree > self.dimension:
            raise SimplicialError("degree is outside the complex")


class _SubsetDomain(Protocol):
    @property
    def dimension(self) -> int: ...

    def simplex_count(self, degree: int) -> int: ...

    def simplices(self, degree: int) -> IndexArray: ...


class _CoefficientSpace(Protocol):
    @property
    def complex(self) -> _SubsetDomain: ...

    @property
    def size(self) -> int: ...

    def same_space(self, other: object) -> bool: ...


class _CochainParent(_CoefficientSpace):
    @property
    def degree(self) -> int:
        raise NotImplementedError


class SimplexSubset[K: _SubsetDomain]:
    """Degree-aligned simplex membership bound to one runtime complex."""

    __slots__ = ("_complex", "_masks")

    def __init__(self, complex_: K, masks: tuple[BoolArray, ...]) -> None:
        if len(masks) != complex_.dimension + 1:
            raise SimplicialError("subset masks must cover every degree")
        owned: list[BoolArray] = []
        for degree, mask in enumerate(masks):
            candidate = np.asarray(mask)
            if candidate.dtype.kind != "b" or candidate.shape != (
                complex_.simplex_count(degree),
            ):
                raise SimplicialError("subset mask does not align with its basis")
            owned.append(_owned_array(candidate, np.dtype(np.bool_)))
        self._complex = complex_
        self._masks = tuple(owned)

    @property
    def complex(self) -> K:
        return self._complex

    def mask(self, degree: int) -> BoolArray:
        self._complex.simplex_count(degree)
        return self._masks[degree].copy()

    def closure(self) -> SimplexSubset[K]:
        return self._select_relation(include_faces=True, include_cofaces=False)

    def star(self) -> SimplexSubset[K]:
        return self._select_relation(include_faces=False, include_cofaces=True)

    def link(self) -> SimplexSubset[K]:
        complex_ = self._complex
        selected = self._selected_keys()
        all_keys = {
            tuple(int(v) for v in row)
            for degree in range(complex_.dimension + 1)
            for row in complex_.simplices(degree)
        }
        masks: list[BoolArray] = []
        for degree in range(complex_.dimension + 1):
            basis = complex_.simplices(degree)
            mask = np.zeros(len(basis), dtype=np.bool_)
            for index, row in enumerate(basis):
                candidate = frozenset(int(v) for v in row)
                mask[index] = any(
                    candidate.isdisjoint(simplex)
                    and tuple(sorted(candidate | simplex)) in all_keys
                    for simplex in selected
                )
            masks.append(mask)
        return SimplexSubset(complex_, tuple(masks))

    def is_pure(self, degree: int) -> bool:
        complex_ = self._complex
        complex_.simplex_count(degree)
        selected = [
            (frozenset(int(v) for v in row), k)
            for k in range(complex_.dimension + 1)
            for row, keep in zip(complex_.simplices(k), self._masks[k], strict=True)
            if keep
        ]
        if not selected:
            return False
        maximal = [
            item_degree
            for index, (simplex, item_degree) in enumerate(selected)
            if not any(
                simplex < other
                for other_index, (other, _) in enumerate(selected)
                if other_index != index
            )
        ]
        return bool(maximal) and all(item_degree == degree for item_degree in maximal)

    def same_members(self, other: SimplexSubset[K]) -> bool:
        if self._complex is not other._complex:
            raise SimplicialError("subsets belong to different complexes")
        return all(
            np.array_equal(left_mask, right_mask)
            for left_mask, right_mask in zip(self._masks, other._masks, strict=True)
        )

    def _selected_keys(self) -> list[frozenset[int]]:
        complex_ = self._complex
        return [
            frozenset(int(v) for v in row)
            for degree in range(complex_.dimension + 1)
            for row, keep in zip(
                complex_.simplices(degree), self._masks[degree], strict=True
            )
            if keep
        ]

    def _select_relation(
        self, *, include_faces: bool, include_cofaces: bool
    ) -> SimplexSubset[K]:
        complex_ = self._complex
        selected = self._selected_keys()
        masks: list[BoolArray] = []
        for degree in range(complex_.dimension + 1):
            basis = complex_.simplices(degree)
            mask = np.zeros(len(basis), dtype=np.bool_)
            for index, row in enumerate(basis):
                candidate = frozenset(int(v) for v in row)
                mask[index] = any(
                    (include_faces and candidate <= simplex)
                    or (include_cofaces and simplex <= candidate)
                    for simplex in selected
                )
            masks.append(mask)
        return SimplexSubset(complex_, tuple(masks))


def topological_boundary[
    B: BoundaryState,
    O: OrientationState,
    C: ConnectivityState,
    T: CodimensionOneRegular,
](
    complex_: Complex[B, O, C, T],
) -> SimplexSubset[Complex[B, O, C, T]]:
    """Return the unsigned, closure-complete ordinary topological boundary."""
    if not isinstance(complex_.topology_state, CodimensionOneRegular):
        raise SimplicialError(
            "topological boundary requires codimension-one regular input"
        )
    evidence = complex_._boundary_evidence
    if evidence is None:
        raise SimplicialError("codimension-one regular state has no verified evidence")
    return SimplexSubset(complex_, evidence._masks)


class CochainSpace[K: _SubsetDomain, Degree: int](_CochainParent):
    __slots__ = ("_complex", "_degree", "_size")

    def __init__(self, complex_: K, degree: Degree) -> None:
        self._complex = complex_
        self._degree = degree
        self._size = complex_.simplex_count(degree)

    @property
    def complex(self) -> K:
        return self._complex

    @property
    def degree(self) -> Degree:
        return self._degree

    @property
    def size(self) -> int:
        return self._size

    def belongs_to(self, complex_: K) -> bool:
        return self._complex is complex_

    def same_space(self, other: object) -> bool:
        return (
            isinstance(other, CochainSpace)
            and self._complex is other._complex
            and self._degree == other._degree
        )

    def form[Semantics: FieldSemantics](
        self, coefficients: CoefficientArray, semantics: Semantics
    ) -> Form[CochainSpace[K, Degree], Semantics]:
        return Form(self, coefficients, semantics)


class CochainSubspace[ParentSpace: _CochainParent]:
    """A canonical coefficient subspace retaining its exact parent space."""

    __slots__ = ("_parent", "_indices")

    def __init__(
        self,
        parent: ParentSpace,
        indices: IndexArray,
    ) -> None:
        if not isinstance(parent, CochainSpace):
            raise SimplicialError("cochain subspace requires a primal parent space")
        candidate = np.asarray(indices)
        if candidate.ndim != 1 or candidate.dtype.kind not in "iu":
            raise SimplicialError(
                "subspace indices must be a one-dimensional integer array"
            )
        if np.any(candidate < 0) or np.any(candidate >= parent.size):
            raise SimplicialError("subspace index is outside the parent space")
        admitted = np.array(candidate, dtype=np.int64, order="C", copy=True)
        if len(admitted) > 1 and np.any(admitted[1:] <= admitted[:-1]):
            raise SimplicialError("subspace indices must be strictly increasing")
        admitted.flags.writeable = False
        self._parent = parent
        self._indices = admitted

    @property
    def parent(self) -> ParentSpace:
        return self._parent

    @property
    def complex(self) -> _SubsetDomain:
        return self._parent.complex

    @property
    def degree(self) -> int:
        return self._parent.degree

    @property
    def size(self) -> int:
        return len(self._indices)

    def indices(self) -> IndexArray:
        return self._indices.copy()

    def belongs_to(self, parent: ParentSpace) -> bool:
        return self._parent is parent

    def same_space(self, other: object) -> bool:
        return (
            isinstance(other, CochainSubspace)
            and self._parent.same_space(other._parent)
            and np.array_equal(self._indices, other._indices)
        )

    def complement(self) -> Self:
        selected = np.ones(self._parent.size, dtype=np.bool_)
        selected[self._indices] = False
        return type(self)(self._parent, np.flatnonzero(selected))

    def form[Semantics: FieldSemantics](
        self, coefficients: CoefficientArray, semantics: Semantics
    ) -> Form[CochainSubspace[ParentSpace], Semantics]:
        return Form(self, coefficients, semantics)


class Form[Space: _CoefficientSpace, Semantics: FieldSemantics]:
    __slots__ = ("_space", "_coefficients", "_semantics")

    def __init__(
        self,
        space: Space,
        coefficients: CoefficientArray,
        semantics: Semantics,
    ) -> None:
        candidate = np.asarray(coefficients)
        if candidate.ndim != 1 or candidate.shape != (space.size,):
            raise SimplicialError("coefficients do not align with the cochain space")
        if np.iscomplexobj(candidate):
            raise SimplicialError("coefficients must be real")
        if candidate.dtype.kind not in "bifu":
            raise SimplicialError("coefficients must be real numeric values")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                admitted = _owned_array(candidate, np.dtype(np.float64))
        except (TypeError, ValueError, OverflowError) as error:
            raise SimplicialError(
                "coefficients cannot be converted to float64"
            ) from error
        if not np.all(np.isfinite(admitted)):
            raise SimplicialError("coefficients must be finite")
        self._space = space
        self._coefficients = admitted
        self._semantics = semantics

    @property
    def space(self) -> Space:
        return self._space

    def coefficients(self) -> CoefficientArray:
        """Return caller-owned coefficients."""
        return self._coefficients.copy()

    @property
    def semantics(self) -> Semantics:
        return self._semantics

    def uses_semantics(self, semantics: Semantics) -> bool:
        return type(self._semantics) is type(semantics)


type ZeroForm[K: _SubsetDomain] = Form[CochainSpace[K, Literal[0]], OrdinaryForm]
type OneForm[K: _SubsetDomain] = Form[CochainSpace[K, Literal[1]], OrdinaryForm]
type TwoForm[K: _SubsetDomain] = Form[CochainSpace[K, Literal[2]], OrdinaryForm]


def _assemble_boundary_matrix(data: _ComplexData, degree: int) -> csr_array:
    columns = len(data._simplices[degree])
    if degree == 0:
        return csr_array((0, columns), dtype=np.int8)
    lower = data._simplices[degree - 1]
    lookup = {tuple(int(v) for v in row): index for index, row in enumerate(lower)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[int] = []
    for column, simplex in enumerate(data._simplices[degree]):
        orientation = int(data._orientations[degree][column])
        key = tuple(int(v) for v in simplex)
        for removed in range(degree + 1):
            face = key[:removed] + key[removed + 1 :]
            row_indices.append(lookup[face])
            column_indices.append(column)
            values.append(orientation * (-1 if removed % 2 else 1))
    return csr_array(
        (
            np.array(values, dtype=np.int8),
            (
                np.array(row_indices, dtype=np.int64),
                np.array(column_indices, dtype=np.int64),
            ),
        ),
        shape=(len(lower), columns),
        dtype=np.int8,
    )


def _require_codimension_one_regular[
    B: BoundaryState,
    O: OrientationState,
    C: ConnectivityState,
](
    complex_: Complex[B, O, C, Simplicial],
) -> tuple[BoolArray, ...]:
    masks = [
        np.zeros(complex_.simplex_count(degree), dtype=np.bool_)
        for degree in range(complex_.dimension + 1)
    ]
    if complex_.dimension == 0:
        return tuple(masks)
    used_vertices = np.unique(complex_._data._simplices[complex_.dimension])
    if len(used_vertices) != complex_.vertex_count:
        raise SimplicialError("codimension-one regular input must be pure")
    coface_counts = np.diff(complex_.boundary_matrix(complex_.dimension).tocsr().indptr)
    if np.any((coface_counts < 1) | (coface_counts > 2)):
        raise SimplicialError(
            "every codimension-one simplex needs one or two top cofaces"
        )
    masks[complex_.dimension - 1] = coface_counts == 1
    for degree in range(complex_.dimension - 1, 0, -1):
        incidence = abs(complex_.boundary_matrix(degree)).astype(np.int64)
        masks[degree - 1] = (
            np.asarray(incidence @ masks[degree].astype(np.int64)).ravel() > 0
        )
    return tuple(masks)


def _require_triangle_manifold[
    B: BoundaryState,
    O: OrientationState,
    C: ConnectivityState,
](
    complex_: Complex[B, O, C, Simplicial],
) -> tuple[BoolArray, ...]:
    if complex_.dimension != 2:
        raise SimplicialError("triangle-manifold refinement requires dimension two")
    boundary_masks = _require_codimension_one_regular(complex_)
    triangles = complex_._data._simplices[2]
    for vertex in range(complex_.vertex_count):
        link: dict[int, set[int]] = {}
        for triangle in triangles:
            values = [int(v) for v in triangle]
            if vertex not in values:
                continue
            other = [value for value in values if value != vertex]
            link.setdefault(other[0], set()).add(other[1])
            link.setdefault(other[1], set()).add(other[0])
        if not link:
            raise SimplicialError("every admitted vertex must have one manifold fan")
        pending = [next(iter(link))]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(link[current] - seen)
        degrees = [len(neighbors) for neighbors in link.values()]
        path = degrees.count(1) == 2 and all(value in (1, 2) for value in degrees)
        cycle = all(value == 2 for value in degrees)
        if len(seen) != len(link) or not (path or cycle):
            raise SimplicialError("a vertex link must be one path or one cycle")
    return boundary_masks


def _require_oriented[B: BoundaryState, C: ConnectivityState, T: TopologyState](
    complex_: Complex[B, OrientationUnknown, C, T],
) -> None:
    if complex_.dimension == 0:
        return
    matrix = complex_.boundary_matrix(complex_.dimension).tocsr()
    for row in range(matrix.shape[0]):
        values = matrix.data[matrix.indptr[row] : matrix.indptr[row + 1]]
        if len(values) > 2 or (len(values) == 2 and int(values.sum()) != 0):
            raise SimplicialError("top-simplex orientations are not coherent")


def _require_boundary_extent[
    O: OrientationState,
    C: ConnectivityState,
    T: TopologyState,
](
    complex_: Complex[BoundaryUnknown, O, C, T],
    *,
    present: bool,
) -> None:
    if not isinstance(complex_.topology_state, CodimensionOneRegular):
        raise SimplicialError(
            "boundary classification requires codimension-one regular input"
        )
    evidence = complex_._boundary_evidence
    if evidence is None:
        raise SimplicialError("codimension-one regular state has no verified evidence")
    actual = complex_.dimension > 0 and bool(
        evidence._masks[complex_.dimension - 1].any()
    )
    if actual != present:
        expected = "nonempty" if present else "empty"
        raise SimplicialError(
            f"the complex does not have {expected} topological boundary"
        )


def _require_connected[B: BoundaryState, O: OrientationState, T: TopologyState](
    complex_: Complex[B, O, ConnectivityUnknown, T],
) -> None:
    vertex_count = complex_.vertex_count
    if vertex_count >= 128 and complex_.dimension >= 1:
        edges = complex_._data._simplices[1]
        directed = np.concatenate((edges, edges[:, ::-1]))
        adjacency = csr_array(
            (
                np.ones(len(directed), dtype=np.bool_),
                (directed[:, 0], directed[:, 1]),
            ),
            shape=(vertex_count, vertex_count),
        )
        connected = (
            csgraph.connected_components(adjacency, directed=False, return_labels=False)
            == 1
        )
    else:
        adjacency = [set[int]() for _ in range(vertex_count)]
        if complex_.dimension >= 1:
            for edge in complex_._data._simplices[1]:
                left, right = (int(edge[0]), int(edge[1]))
                adjacency[left].add(right)
                adjacency[right].add(left)
        pending, seen = [0], set[int]()
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(adjacency[current] - seen)
        connected = len(seen) == vertex_count
    if not connected:
        raise SimplicialError("the complex is disconnected")


__all__ = [
    "ORDINARY_FORM",
    "BoundaryState",
    "BoundaryUnknown",
    "CochainSpace",
    "CochainSubspace",
    "CodimensionOneRegular",
    "Complex",
    "Connected",
    "ConnectivityState",
    "ConnectivityUnknown",
    "FieldSemantics",
    "Form",
    "OneForm",
    "OrdinaryForm",
    "OrientationState",
    "OrientationUnknown",
    "Oriented",
    "SimplexSubset",
    "Simplicial",
    "SimplicialError",
    "TopologyState",
    "TriangleManifold",
    "TwoForm",
    "WithBoundary",
    "WithoutBoundary",
    "ZeroForm",
    "topological_boundary",
]
