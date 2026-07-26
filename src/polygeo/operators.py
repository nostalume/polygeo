"""Typed linear maps over exact PolyGeo cochain spaces."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from .geometry import Geometry, GeometryError, _GeometryDomain
from .simplicial import CochainSpace, FieldSemantics, Form, _CoefficientSpace


class OperatorError(ValueError):
    """Invalid linear-map construction or application."""


class _OperatorDomain(Protocol):
    @property
    def dimension(self) -> int: ...

    def simplex_count(self, degree: int) -> int: ...

    def simplices(self, degree: int) -> NDArray[np.int64]: ...

    def boundary_matrix(self, degree: int) -> csr_array: ...


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
    try:
        primal = geometry.primal_measures(source.degree)
        dual = geometry.dual_measures(source.degree)
    except GeometryError as error:
        raise OperatorError("Hodge measures are not representable") from error
    with np.errstate(all="ignore"):
        weights = np.divide(dual, primal)
    if not np.all(np.isfinite(weights)) or np.any((dual != 0.0) & (weights == 0.0)):
        raise OperatorError("Hodge coefficients are not representable as float64")
    target = DualCochainSpace(geometry, source)
    indices = np.arange(source.size, dtype=np.int64)
    matrix = csr_array(
        (weights, (indices, indices)),
        shape=(target.size, source.size),
    )
    return LinearMap(source, target, matrix)


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
    with np.errstate(over="ignore", invalid="ignore"):
        owned.sum_duplicates()
    owned.eliminate_zeros()
    owned.sort_indices()
    if not np.all(np.isfinite(owned.data)):
        raise OperatorError("linear-map coefficients must be finite")
    return owned


__all__ = [
    "DualCochainSpace",
    "LinearMap",
    "OperatorError",
    "exterior_derivative",
    "hodge_star",
]
