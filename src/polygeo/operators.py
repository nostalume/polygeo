"""Typed linear maps over exact PolyGeo cochain spaces."""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from .geometry import Geometry, GeometryError, _GeometryDomain
from .simplicial import (
    CochainSpace,
    CochainSubspace,
    FieldSemantics,
    Form,
    _CochainParent,
    _CoefficientSpace,
)


class OperatorError(ValueError):
    """Invalid linear-map construction or application."""


class _OperatorDomain(Protocol):
    @property
    def dimension(self) -> int: ...

    def simplex_count(self, degree: int) -> int: ...

    def simplices(self, degree: int) -> NDArray[np.int64]: ...

    def boundary_matrix(self, degree: int) -> csr_array: ...


class _MetricOperatorDomain(_GeometryDomain, _OperatorDomain, Protocol):
    pass


class DualCochainSpace[
    K: _GeometryDomain,
    PrimalDegree: int,
]:
    """Dual cochains subordinate to one exact geometry and primal space."""

    __slots__ = ("_geometry", "_primal")
    _geometry: Geometry[K]
    _primal: CochainSpace[K, PrimalDegree]

    def __init__(
        self,
        geometry: Geometry[K],
        primal: CochainSpace[K, PrimalDegree],
    ) -> None:
        if geometry.complex is not primal.complex:
            raise OperatorError(
                "dual space geometry and primal belong to different complexes"
            )
        self._geometry = geometry
        self._primal = primal

    @property
    def geometry(self) -> Geometry[K]:
        return self._geometry

    @property
    def primal(self) -> CochainSpace[K, PrimalDegree]:
        return self._primal

    @property
    def complex(self) -> K:
        return self._primal.complex

    @property
    def primal_degree(self) -> PrimalDegree:
        return self._primal.degree

    @property
    def degree(self) -> int:
        return self._primal.complex.dimension - self._primal.degree

    @property
    def size(self) -> int:
        return self._primal.size

    def same_space(self, other: object) -> bool:
        return (
            isinstance(other, DualCochainSpace)
            and self._geometry is other._geometry
            and self._primal.same_space(other._primal)
        )


class LinearMap[
    SourceSpace: _CoefficientSpace,
    TargetSpace: _CoefficientSpace,
]:
    """Complete linear map between exact runtime coefficient spaces."""

    __slots__ = ("_matrix", "_source", "_target")
    _matrix: csr_array
    _source: SourceSpace
    _target: TargetSpace

    def __init__(
        self,
        source: SourceSpace,
        target: TargetSpace,
        matrix: csr_array,
    ) -> None:
        if source.complex is not target.complex:
            raise OperatorError("map spaces belong to different complexes")
        admitted = _admit_matrix(source, target, matrix)
        self._source = source
        self._target = target
        self._matrix = admitted

    @property
    def source(self) -> SourceSpace:
        return self._source

    @property
    def target(self) -> TargetSpace:
        return self._target

    def matrix(self) -> csr_array:
        """Return a caller-owned matrix representation."""
        return self._matrix.copy()

    def apply[Semantics: FieldSemantics](
        self,
        value: Form[SourceSpace, Semantics],
    ) -> Form[TargetSpace, Semantics]:
        if not value.space.same_space(self._source):
            raise OperatorError("form does not belong to the map source")
        with np.errstate(over="ignore", invalid="ignore"):
            coefficients = np.asarray(
                self._matrix @ value._coefficients,
                dtype=np.float64,
            ).reshape(-1)
        if not np.all(np.isfinite(coefficients)):
            raise OperatorError("map application produced non-finite coefficients")
        return Form(self._target, coefficients, value.semantics)

    def compose[InputSpace: _CoefficientSpace](
        self,
        before: LinearMap[InputSpace, SourceSpace],
    ) -> LinearMap[InputSpace, TargetSpace]:
        """Return this map composed after a compatible map."""
        if not before.target.same_space(self._source):
            raise OperatorError("composed maps have different intermediate spaces")
        return LinearMap(
            before.source,
            self._target,
            self._matrix @ before._matrix,
        )


def exterior_derivative[
    K: _OperatorDomain,
    SourceDegree: int,
    TargetDegree: int,
](
    source: CochainSpace[K, SourceDegree],
    target: CochainSpace[K, TargetDegree],
) -> LinearMap[
    CochainSpace[K, SourceDegree],
    CochainSpace[K, TargetDegree],
]:
    """Construct the cochain differential between adjacent exact spaces."""
    if source.complex is not target.complex:
        raise OperatorError("map spaces belong to different complexes")
    if target.degree != source.degree + 1:
        raise OperatorError("exterior derivative requires adjacent cochain degrees")
    matrix = source.complex.boundary_matrix(target.degree).transpose().tocsr()
    return LinearMap(source, target, matrix)


def restrict[ParentSpace: _CochainParent](
    parent: ParentSpace,
    subspace: CochainSubspace[ParentSpace],
) -> LinearMap[ParentSpace, CochainSubspace[ParentSpace]]:
    """Select canonical parent coefficients without orientation conversion."""
    if not subspace.belongs_to(parent):
        raise OperatorError("restriction requires the subspace's exact parent")
    rows = np.arange(subspace.size, dtype=np.int64)
    indices = subspace.indices()
    matrix = csr_array(
        (np.ones(subspace.size), (rows, indices)),
        shape=(subspace.size, parent.size),
    )
    return LinearMap(parent, subspace, matrix)


def extend_zero[ParentSpace: _CochainParent](
    subspace: CochainSubspace[ParentSpace],
    parent: ParentSpace,
) -> LinearMap[CochainSubspace[ParentSpace], ParentSpace]:
    """Insert subspace coefficients into their parent and zero the complement."""
    if not subspace.belongs_to(parent):
        raise OperatorError("zero extension requires the subspace's exact parent")
    columns = np.arange(subspace.size, dtype=np.int64)
    indices = subspace.indices()
    matrix = csr_array(
        (np.ones(subspace.size), (indices, columns)),
        shape=(parent.size, subspace.size),
    )
    return LinearMap(subspace, parent, matrix)


def hodge_star[
    K: _GeometryDomain,
    Degree: int,
](
    geometry: Geometry[K],
    source: CochainSpace[K, Degree],
) -> LinearMap[
    CochainSpace[K, Degree],
    DualCochainSpace[K, Degree],
]:
    """Construct the signed circumcentric Hodge star on one primal space."""
    if geometry.complex is not source.complex:
        raise OperatorError("Hodge geometry and source belong to different complexes")
    weights = _hodge_weights(geometry, source.degree)
    target = DualCochainSpace(geometry, source)
    indices = np.arange(source.size, dtype=np.int64)
    matrix = csr_array(
        (weights, (indices, indices)),
        shape=(target.size, source.size),
    )
    return LinearMap(source, target, matrix)


def weighted_pairing[
    K: _GeometryDomain,
    Degree: int,
    LeftSemantics: FieldSemantics,
    RightSemantics: FieldSemantics,
](
    geometry: Geometry[K],
    left: Form[CochainSpace[K, Degree], LeftSemantics],
    right: Form[CochainSpace[K, Degree], RightSemantics],
) -> float:
    """Evaluate the signed circumcentric weighted bilinear pairing."""
    if geometry.complex is not left.space.complex:
        raise OperatorError("pairing geometry and forms belong to different complexes")
    if not left.space.same_space(right.space):
        raise OperatorError("pairing forms must share the same cochain space")
    weights = _hodge_weights(geometry, left.space.degree)
    with np.errstate(all="ignore"):
        intermediate = left._coefficients * weights
        terms = intermediate * right._coefficients
    nonzero = (
        (left._coefficients != 0.0) & (weights != 0.0) & (right._coefficients != 0.0)
    )
    tiny = np.finfo(np.float64).tiny
    suspicious_intermediate = (
        ~np.isfinite(intermediate)
        | ((intermediate == 0.0) & (left._coefficients != 0.0) & (weights != 0.0))
        | ((intermediate != 0.0) & (np.abs(intermediate) < tiny))
    )
    suspicious_term = (
        ~np.isfinite(terms)
        | (nonzero & (terms == 0.0))
        | ((terms != 0.0) & (np.abs(terms) < tiny))
    )
    if np.any(suspicious_intermediate) or np.any(suspicious_term):
        return _exact_pairing(left._coefficients, weights, right._coefficients)
    try:
        value = math.fsum(float(term) for term in terms)
    except OverflowError:
        return _exact_pairing(left._coefficients, weights, right._coefficients)
    if value == 0.0 and np.any(terms != 0.0):
        return _exact_pairing(left._coefficients, weights, right._coefficients)
    if not math.isfinite(value):
        raise OperatorError("weighted pairing is not representable as float64")
    return value


def codifferential[
    K: _GeometryDomain,
    PreviousDegree: int,
    Degree: int,
](
    geometry: Geometry[K],
    derivative: LinearMap[
        CochainSpace[K, PreviousDegree],
        CochainSpace[K, Degree],
    ],
) -> LinearMap[
    CochainSpace[K, Degree],
    CochainSpace[K, PreviousDegree],
]:
    """Construct the weighted adjoint of an adjacent primal cochain map."""
    if not isinstance(derivative.source, CochainSpace) or not isinstance(
        derivative.target, CochainSpace
    ):
        raise OperatorError("codifferential requires primal cochain map endpoints")
    if geometry.complex is not derivative.source.complex:
        raise OperatorError(
            "codifferential geometry and map belong to different complexes"
        )
    if derivative.target.degree != derivative.source.degree + 1:
        raise OperatorError("codifferential requires adjacent degrees")
    matrix = _codifferential_matrix(
        derivative._matrix,
        _hodge_weights(geometry, derivative.source.degree),
        _hodge_weights(geometry, derivative.target.degree),
    )
    return LinearMap(derivative.target, derivative.source, matrix)


def hodge_laplacian[
    K: _MetricOperatorDomain,
    Degree: int,
](
    geometry: Geometry[K],
    space: CochainSpace[K, Degree],
) -> LinearMap[
    CochainSpace[K, Degree],
    CochainSpace[K, Degree],
]:
    """Construct the degree-wise signed circumcentric Hodge Laplacian."""
    if geometry.complex is not space.complex:
        raise OperatorError(
            "Laplacian geometry and space belong to different complexes"
        )
    degree = space.degree
    current_weights = _hodge_weights(geometry, degree)
    terms: list[csr_array] = []
    if degree > 0:
        lower = space.complex.boundary_matrix(degree).transpose().tocsr()
        delta = _codifferential_matrix(
            lower,
            _hodge_weights(geometry, degree - 1),
            current_weights,
        )
        with np.errstate(all="ignore"):
            terms.append((lower @ delta).tocsr())
    if degree < space.complex.dimension:
        upper = space.complex.boundary_matrix(degree + 1).transpose().tocsr()
        delta = _codifferential_matrix(
            upper,
            current_weights,
            _hodge_weights(geometry, degree + 1),
        )
        with np.errstate(all="ignore"):
            terms.append((delta @ upper).tocsr())
    if not terms:
        matrix = csr_array((space.size, space.size), dtype=np.float64)
    elif len(terms) == 1:
        matrix = terms[0]
    else:
        with np.errstate(all="ignore"):
            matrix = (terms[0] + terms[1]).tocsr()
    return LinearMap(space, space, matrix)


def _hodge_weights[K: _GeometryDomain](
    geometry: Geometry[K],
    degree: int,
) -> NDArray[np.float64]:
    try:
        primal = geometry.primal_measures(degree)
        dual = geometry.dual_measures(degree)
    except GeometryError as error:
        raise OperatorError("Hodge measures are not representable") from error
    with np.errstate(all="ignore"):
        weights = np.divide(dual, primal)
    if not np.all(np.isfinite(weights)) or np.any((dual != 0.0) & (weights == 0.0)):
        raise OperatorError("Hodge coefficients are not representable as float64")
    return weights


def _exact_pairing(
    left: NDArray[np.float64],
    weights: NDArray[np.float64],
    right: NDArray[np.float64],
) -> float:
    exact = sum(
        (
            Fraction(float(left_value))
            * Fraction(float(weight))
            * Fraction(float(right_value))
            for left_value, weight, right_value in zip(
                left, weights, right, strict=True
            )
        ),
        start=Fraction(),
    )
    try:
        value = float(exact)
    except OverflowError as error:
        raise OperatorError(
            "weighted pairing is not representable as float64"
        ) from error
    if not math.isfinite(value) or (value == 0.0 and exact != 0):
        raise OperatorError("weighted pairing is not representable as float64")
    return value


def _codifferential_matrix(
    derivative: csr_array,
    previous_weights: NDArray[np.float64],
    current_weights: NDArray[np.float64],
) -> csr_array:
    if np.any(previous_weights == 0.0):
        raise OperatorError("codifferential has a zero reciprocal Hodge weight")
    matrix = derivative.transpose().tocsr(copy=True)
    column_weights = current_weights[matrix.indices]
    row_weights = np.repeat(previous_weights, np.diff(matrix.indptr))
    nonzero = (matrix.data != 0.0) & (column_weights != 0.0) & (row_weights != 0.0)
    with np.errstate(all="ignore"):
        quotient = matrix.data / row_weights
        scaled = quotient * column_weights
    tiny = np.finfo(np.float64).tiny
    suspicious = (
        ~np.isfinite(quotient)
        | ((quotient == 0.0) & (matrix.data != 0.0))
        | ((quotient != 0.0) & (np.abs(quotient) < tiny))
        | ~np.isfinite(scaled)
        | (nonzero & (scaled == 0.0))
        | ((scaled != 0.0) & (np.abs(scaled) < tiny))
    )
    for index in np.flatnonzero(suspicious):
        exact = (
            Fraction(float(matrix.data[index]))
            * Fraction(float(column_weights[index]))
            / Fraction(float(row_weights[index]))
        )
        try:
            scaled[index] = float(exact)
        except OverflowError as error:
            raise OperatorError(
                "codifferential coefficients are not representable as float64"
            ) from error
        if not np.isfinite(scaled[index]) or (scaled[index] == 0.0 and exact != 0):
            raise OperatorError(
                "codifferential coefficients are not representable as float64"
            )
    matrix.data = scaled
    matrix.eliminate_zeros()
    return matrix


def _admit_matrix[
    SourceSpace: _CoefficientSpace,
    TargetSpace: _CoefficientSpace,
](
    source: SourceSpace,
    target: TargetSpace,
    matrix: csr_array,
) -> csr_array:
    if not isinstance(matrix, csr_array):
        raise OperatorError("linear-map representation must be a CSR sparse array")
    supported_index_dtypes = (np.dtype(np.int32), np.dtype(np.int64))
    if (
        matrix.indices.dtype not in supported_index_dtypes
        or matrix.indptr.dtype not in supported_index_dtypes
    ):
        raise OperatorError("linear-map representation has invalid CSR structure")
    try:
        matrix.check_format(full_check=True)
    except ValueError as error:
        raise OperatorError(
            "linear-map representation has invalid CSR structure"
        ) from error
    if matrix.shape != (target.size, source.size):
        raise OperatorError("linear-map shape does not match its spaces")
    if np.iscomplexobj(matrix.data):
        raise OperatorError("linear-map coefficients must be real")
    try:
        owned = csr_array(matrix, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise OperatorError(
            "linear-map representation cannot be converted to float64"
        ) from error
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            owned.sum_duplicates()
        owned.eliminate_zeros()
        owned.sort_indices()
    except (IndexError, ValueError, OverflowError) as error:
        raise OperatorError(
            "linear-map representation has invalid CSR structure"
        ) from error
    if not np.all(np.isfinite(owned.data)):
        raise OperatorError("linear-map coefficients must be finite")
    return owned


__all__ = [
    "codifferential",
    "DualCochainSpace",
    "hodge_laplacian",
    "LinearMap",
    "OperatorError",
    "extend_zero",
    "exterior_derivative",
    "hodge_star",
    "restrict",
    "weighted_pairing",
]
