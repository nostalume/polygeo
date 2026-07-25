"""Typed linear maps over exact PolyGeo cochain spaces."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from .simplicial import CochainSpace, FieldSemantics, Form


class OperatorError(ValueError):
    """Invalid linear-map construction or application."""


class _OperatorDomain(Protocol):
    @property
    def dimension(self) -> int: ...

    def simplex_count(self, degree: int) -> int: ...

    def simplices(self, degree: int) -> NDArray[np.int64]: ...

    def boundary_matrix(self, degree: int) -> csr_array: ...


class LinearMap[
    K: _OperatorDomain,
    SourceDegree: int,
    TargetDegree: int,
]:
    """Complete linear map between exact runtime cochain spaces."""

    __slots__ = ("_matrix", "_source", "_target")
    _matrix: csr_array
    _source: CochainSpace[K, SourceDegree]
    _target: CochainSpace[K, TargetDegree]

    def __init__(
        self,
        source: CochainSpace[K, SourceDegree],
        target: CochainSpace[K, TargetDegree],
        matrix: csr_array,
    ) -> None:
        if source.complex is not target.complex:
            raise OperatorError("map spaces belong to different complexes")
        admitted = _admit_matrix(source, target, matrix)
        self._source = source
        self._target = target
        self._matrix = admitted

    @property
    def source(self) -> CochainSpace[K, SourceDegree]:
        return self._source

    @property
    def target(self) -> CochainSpace[K, TargetDegree]:
        return self._target

    def matrix(self) -> csr_array:
        """Return a caller-owned matrix representation."""
        return self._matrix.copy()

    def apply[Semantics: FieldSemantics](
        self,
        value: Form[K, SourceDegree, Semantics],
    ) -> Form[K, TargetDegree, Semantics]:
        if not value.space.same_space(self._source):
            raise OperatorError("form does not belong to the map source")
        with np.errstate(over="ignore", invalid="ignore"):
            coefficients = np.asarray(
                self._matrix @ value._coefficients,
                dtype=np.float64,
            ).reshape(-1)
        if not np.all(np.isfinite(coefficients)):
            raise OperatorError("map application produced non-finite coefficients")
        return self._target.form(coefficients, value.semantics)

    def compose[InputDegree: int](
        self,
        before: LinearMap[K, InputDegree, SourceDegree],
    ) -> LinearMap[K, InputDegree, TargetDegree]:
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
) -> LinearMap[K, SourceDegree, TargetDegree]:
    """Construct the cochain differential between adjacent exact spaces."""
    if source.complex is not target.complex:
        raise OperatorError("map spaces belong to different complexes")
    if target.degree != source.degree + 1:
        raise OperatorError("exterior derivative requires adjacent cochain degrees")
    matrix = source.complex.boundary_matrix(target.degree).transpose().tocsr()
    return LinearMap(source, target, matrix)


def _admit_matrix[
    K: _OperatorDomain,
    SourceDegree: int,
    TargetDegree: int,
](
    source: CochainSpace[K, SourceDegree],
    target: CochainSpace[K, TargetDegree],
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


__all__ = ["LinearMap", "OperatorError", "exterior_derivative"]
