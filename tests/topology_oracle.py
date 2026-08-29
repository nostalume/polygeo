"""Independent exact set-theoretic oracle for topology checks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

type IndexArray = NDArray[np.int64]
type SignArray = NDArray[np.int8]
type BoolArray = NDArray[np.bool_]


class OracleError(ValueError):
    """Classified failure from the frozen Python topology semantics."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _sign(row: IndexArray) -> np.int8:
    inversions = sum(
        int(row[left] > row[right])
        for left in range(len(row))
        for right in range(left + 1, len(row))
    )
    return np.int8(-1 if inversions % 2 else 1)


def _boundary(
    bases: tuple[IndexArray, ...], orientations: tuple[SignArray, ...], degree: int
) -> csr_array:
    columns = len(bases[degree])
    if degree == 0:
        return csr_array((0, columns), dtype=np.int8)
    lower = bases[degree - 1]
    lookup = {
        tuple(int(value) for value in row): index for index, row in enumerate(lower)
    }
    rows: list[int] = []
    columns_out: list[int] = []
    values: list[int] = []
    for column, (simplex, orientation) in enumerate(
        zip(bases[degree], orientations[degree], strict=True)
    ):
        for removed in range(degree + 1):
            face = tuple(int(value) for value in np.delete(simplex, removed))
            rows.append(lookup[face])
            columns_out.append(column)
            values.append(int(orientation) * (-1 if removed % 2 else 1))
    return csr_array(
        (
            np.asarray(values, dtype=np.int8),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(columns_out, dtype=np.int64),
            ),
        ),
        shape=(len(lower), columns),
        dtype=np.int8,
    )


@dataclass(frozen=True)
class OracleComplex:
    """Canonical exact topology independent of the future public implementation."""

    vertex_count: int
    dimension: int
    simplices: tuple[IndexArray, ...]
    orientations: tuple[SignArray, ...]
    boundaries: tuple[csr_array, ...]

    def regular_boundary(self) -> tuple[BoolArray, ...]:
        masks = [np.zeros(len(basis), dtype=np.bool_) for basis in self.simplices]
        if self.dimension == 0:
            return tuple(masks)
        used = np.unique(self.simplices[self.dimension])
        if len(used) != self.vertex_count:
            raise OracleError("not_pure", "codimension-one regular input must be pure")
        counts = np.diff(self.boundaries[self.dimension].indptr)
        if np.any((counts < 1) | (counts > 2)):
            raise OracleError(
                "codimension_one_incidence",
                "every codimension-one simplex needs one or two top cofaces",
            )
        masks[self.dimension - 1] = counts == 1
        for degree in range(self.dimension - 1, 0, -1):
            incidence = abs(self.boundaries[degree]).astype(np.int64)
            masks[degree - 1] = (
                np.asarray(incidence @ masks[degree].astype(np.int64)).ravel() > 0
            )
        return tuple(masks)

    def triangle_boundary(self) -> tuple[BoolArray, ...]:
        if self.dimension != 2:
            raise OracleError(
                "triangle_dimension",
                "triangle-manifold refinement requires dimension two",
            )
        masks = self.regular_boundary()
        for vertex in range(self.vertex_count):
            link: dict[int, set[int]] = {}
            for triangle in self.simplices[2]:
                values = [int(value) for value in triangle]
                if vertex not in values:
                    continue
                other = [value for value in values if value != vertex]
                link.setdefault(other[0], set()).add(other[1])
                link.setdefault(other[1], set()).add(other[0])
            if not link:
                raise OracleError(
                    "not_pure", "every admitted vertex must have one manifold fan"
                )
            pending = [next(iter(link))]
            seen: set[int] = set()
            while pending:
                current = pending.pop()
                if current not in seen:
                    seen.add(current)
                    pending.extend(link[current] - seen)
            degrees = [len(neighbors) for neighbors in link.values()]
            path = degrees.count(1) == 2 and all(value in (1, 2) for value in degrees)
            cycle = all(value == 2 for value in degrees)
            if len(seen) != len(link) or not (path or cycle):
                raise OracleError(
                    "vertex_link", "a vertex link must be one path or one cycle"
                )
        return masks

    def require_oriented(self) -> None:
        if self.dimension == 0:
            return
        matrix = self.boundaries[self.dimension]
        for row in range(matrix.shape[0]):
            values = matrix.data[matrix.indptr[row] : matrix.indptr[row + 1]]
            if len(values) > 2 or (len(values) == 2 and int(values.sum()) != 0):
                raise OracleError(
                    "orientation", "top-simplex orientations are not coherent"
                )

    def require_connected(self) -> None:
        adjacency = [set[int]() for _ in range(self.vertex_count)]
        if self.dimension >= 1:
            for edge in self.simplices[1]:
                left, right = (int(edge[0]), int(edge[1]))
                adjacency[left].add(right)
                adjacency[right].add(left)
        pending = [0]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(adjacency[current] - seen)
        if len(seen) != self.vertex_count:
            raise OracleError("disconnected", "the complex is disconnected")

    def subset(self, masks: tuple[BoolArray, ...]) -> OracleSubset:
        return OracleSubset(self, masks)


def admit_oracle(
    maximal_simplices: NDArray[np.generic], *, vertex_count: int | None = None
) -> OracleComplex:
    source = np.asarray(maximal_simplices)
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise OracleError(
            "empty_maximal", "maximal simplices must be a nonempty matrix"
        )
    if source.dtype.kind not in "iu" or source.dtype.kind == "b":
        raise OracleError("unsupported_dtype", "vertex indices must be integers")
    rows = np.array(source, dtype=np.int64, order="C", copy=True)
    if np.any(rows < 0):
        raise OracleError("negative_index", "vertex indices must be nonnegative")
    for row in rows:
        if len(np.unique(row)) != len(row):
            raise OracleError("repeated_vertex", "a simplex cannot repeat a vertex")
    canonical = np.sort(rows, axis=1)
    identities = [tuple(int(value) for value in row) for row in canonical]
    if len(set(identities)) != len(identities):
        raise OracleError("duplicate_maximal", "duplicate maximal simplex identity")
    maximum = int(rows.max())
    admitted_vertices = maximum + 1 if vertex_count is None else vertex_count
    if admitted_vertices <= maximum or admitted_vertices < 1:
        raise OracleError("vertex_extent", "vertex_count does not contain every index")
    dimension = rows.shape[1] - 1
    bases: list[IndexArray] = []
    for degree in range(dimension + 1):
        keys = (
            [(vertex,) for vertex in range(admitted_vertices)]
            if degree == 0
            else sorted(
                {
                    face
                    for maximal in identities
                    for face in combinations(maximal, degree + 1)
                }
            )
        )
        bases.append(np.asarray(keys, dtype=np.int64).reshape(-1, degree + 1))
    top_signs = {
        identity: _sign(row) for identity, row in zip(identities, rows, strict=True)
    }
    orientations: list[SignArray] = [
        np.ones(len(basis), dtype=np.int8) for basis in bases
    ]
    orientations[dimension] = np.asarray(
        [top_signs[tuple(int(value) for value in row)] for row in bases[dimension]],
        dtype=np.int8,
    )
    basis_tuple = tuple(bases)
    orientation_tuple = tuple(orientations)
    boundaries = tuple(
        _boundary(basis_tuple, orientation_tuple, degree)
        for degree in range(dimension + 1)
    )
    return OracleComplex(
        admitted_vertices,
        dimension,
        basis_tuple,
        orientation_tuple,
        boundaries,
    )


@dataclass(frozen=True)
class OracleSubset:
    """Frozen set-based relation truth; deliberately not production-efficient."""

    complex: OracleComplex
    masks: tuple[BoolArray, ...]

    def __init__(self, complex_: OracleComplex, masks: tuple[BoolArray, ...]) -> None:
        if len(masks) != complex_.dimension + 1:
            raise OracleError("mask_shape", "subset masks must cover every degree")
        admitted: list[BoolArray] = []
        for degree, mask in enumerate(masks):
            candidate = np.asarray(mask)
            if candidate.dtype.kind != "b" or candidate.shape != (
                len(complex_.simplices[degree]),
            ):
                raise OracleError(
                    "mask_shape", "subset mask does not align with its basis"
                )
            admitted.append(np.array(candidate, dtype=np.bool_, copy=True))
        object.__setattr__(self, "complex", complex_)
        object.__setattr__(self, "masks", tuple(admitted))

    def _selected(self) -> list[frozenset[int]]:
        return [
            frozenset(int(value) for value in row)
            for degree in range(self.complex.dimension + 1)
            for row, keep in zip(
                self.complex.simplices[degree], self.masks[degree], strict=True
            )
            if keep
        ]

    def _relation(self, *, faces: bool, cofaces: bool) -> OracleSubset:
        selected = self._selected()
        masks: list[BoolArray] = []
        for basis in self.complex.simplices:
            output = np.zeros(len(basis), dtype=np.bool_)
            for index, row in enumerate(basis):
                candidate = frozenset(int(value) for value in row)
                output[index] = any(
                    (faces and candidate <= simplex)
                    or (cofaces and simplex <= candidate)
                    for simplex in selected
                )
            masks.append(output)
        return OracleSubset(self.complex, tuple(masks))

    def closure(self) -> OracleSubset:
        return self._relation(faces=True, cofaces=False)

    def star(self) -> OracleSubset:
        return self._relation(faces=False, cofaces=True)

    def link(self) -> OracleSubset:
        selected = self._selected()
        all_keys = {
            tuple(int(value) for value in row)
            for basis in self.complex.simplices
            for row in basis
        }
        masks: list[BoolArray] = []
        for basis in self.complex.simplices:
            output = np.zeros(len(basis), dtype=np.bool_)
            for index, row in enumerate(basis):
                candidate = frozenset(int(value) for value in row)
                output[index] = any(
                    candidate.isdisjoint(simplex)
                    and tuple(sorted(candidate | simplex)) in all_keys
                    for simplex in selected
                )
            masks.append(output)
        return OracleSubset(self.complex, tuple(masks))

    def is_pure(self, degree: int) -> bool:
        if degree < 0 or degree > self.complex.dimension:
            raise OracleError("degree_outside", "degree is outside the complex")
        selected = [
            (frozenset(int(value) for value in row), item_degree)
            for item_degree, basis in enumerate(self.complex.simplices)
            for row, keep in zip(basis, self.masks[item_degree], strict=True)
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
        return bool(maximal) and all(value == degree for value in maximal)
