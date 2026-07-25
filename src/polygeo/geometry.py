"""Complete Euclidean geometry for finite simplicial complexes."""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .simplicial import (
    BoundaryState,
    Complex,
    ConnectivityState,
    OrientationState,
    TopologyState,
)

type FloatArray = NDArray[np.float64]


class GeometryError(ValueError):
    """Invalid, degenerate, or unrepresentable Euclidean geometry."""


class _GeometryDomain(Protocol):
    @property
    def vertex_count(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def simplex_count(self, degree: int) -> int: ...

    def simplices(self, degree: int) -> NDArray[np.int64]: ...


class Geometry[K: _GeometryDomain]:
    """Complete Euclidean geometry bound to one exact simplicial complex."""

    __slots__ = ("_complex", "_measures", "_positions")
    _complex: K
    _measures: tuple[FloatArray, ...]
    _positions: FloatArray

    def __init__[
        B: BoundaryState,
        O: OrientationState,
        C: ConnectivityState,
        T: TopologyState,
    ](
        self: Geometry[Complex[B, O, C, T]],
        complex_: Complex[B, O, C, T],
        positions: FloatArray,
    ) -> None:
        admitted = _admit_positions(complex_, positions)
        measures = _compute_measures(complex_, admitted)
        self._complex = complex_
        self._positions = _owned_float(admitted)
        self._measures = tuple(_owned_float(values) for values in measures)

    @staticmethod
    def from_positions[
        B: BoundaryState,
        O: OrientationState,
        C: ConnectivityState,
        T: TopologyState,
    ](
        complex_: Complex[B, O, C, T],
        positions: FloatArray,
    ) -> Geometry[Complex[B, O, C, T]]:
        return Geometry(complex_, positions)

    @property
    def complex(self) -> K:
        return self._complex

    @property
    def ambient_dimension(self) -> int:
        return self._positions.shape[1]

    @property
    def positions(self) -> FloatArray:
        return self._positions.copy()

    def simplex_measures(self, degree: int) -> FloatArray:
        if degree < 0 or degree > self._complex.dimension:
            raise GeometryError("measure degree is outside the complex")
        return self._measures[degree].copy()


def _owned_float(values: FloatArray) -> FloatArray:
    owned = np.array(values, dtype=np.float64, order="C", copy=True)
    owned.flags.writeable = False
    return owned


def _admit_positions(
    complex_: _GeometryDomain,
    positions: FloatArray,
) -> FloatArray:
    if not isinstance(positions, np.ndarray):
        raise GeometryError("positions must be a float64 ndarray")
    candidate = positions
    if candidate.ndim != 2 or candidate.shape[0] != complex_.vertex_count:
        raise GeometryError("positions must have one row per admitted vertex")
    if candidate.dtype != np.dtype(np.float64):
        raise GeometryError("positions must use float64")
    ambient_dimension = candidate.shape[1]
    if ambient_dimension < complex_.dimension:
        raise GeometryError("ambient dimension must contain the simplicial dimension")
    if not np.all(np.isfinite(candidate)):
        raise GeometryError("positions must be finite")
    return candidate


def _compute_measures(
    complex_: _GeometryDomain,
    positions: FloatArray,
) -> tuple[FloatArray, ...]:
    measures: list[FloatArray] = [np.ones(complex_.simplex_count(0), dtype=np.float64)]
    for degree in range(1, complex_.dimension + 1):
        basis = complex_.simplices(degree)
        values = np.empty(len(basis), dtype=np.float64)
        for index, simplex in enumerate(basis):
            values[index] = _simplex_measure(positions[simplex], degree)
        measures.append(values)
    return tuple(measures)


def _simplex_measure(points: FloatArray, degree: int) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        edges = (points[1:] - points[0]).T
    if not np.all(np.isfinite(edges)):
        raise GeometryError("simplex differences are not representable")

    scales = np.max(np.abs(edges), axis=0)
    if np.any(scales == 0.0) or not np.all(np.isfinite(scales)):
        raise GeometryError("simplex is degenerate")

    normalized = edges / scales
    try:
        _, triangular = np.linalg.qr(normalized, mode="reduced")
    except np.linalg.LinAlgError as error:
        raise GeometryError("simplex factorization failed") from error

    diagonal = np.abs(np.diag(triangular))
    if len(diagonal) != degree:
        raise GeometryError("simplex factorization has the wrong rank")
    suspicion = (
        np.finfo(np.float64).eps
        * max(normalized.shape)
        * float(np.max(diagonal))
        * 16.0
    )
    if float(np.min(diagonal)) <= suspicion:
        return _exact_measure(points, degree)
    if np.any(diagonal == 0.0) or not np.all(np.isfinite(diagonal)):
        raise GeometryError("simplex measure is not representable")

    return _scaled_measure(scales, diagonal, degree)


def _exact_measure(points: FloatArray, degree: int) -> float:
    exact_points = [[Fraction(float(value)) for value in point] for point in points]
    exact_edges = [
        [
            exact_points[vertex][coordinate] - exact_points[0][coordinate]
            for vertex in range(1, degree + 1)
        ]
        for coordinate in range(points.shape[1])
    ]
    gram = [
        [
            sum(
                (row[left] * row[right] for row in exact_edges),
                start=Fraction(),
            )
            for right in range(degree)
        ]
        for left in range(degree)
    ]
    squared_volume = _determinant(gram)
    if squared_volume <= 0:
        raise GeometryError("simplex is degenerate")

    try:
        with localcontext() as context:
            context.prec = max(50, degree * 16)
            context.Emax = 999_999_999
            context.Emin = -999_999_999
            root = (
                Decimal(squared_volume.numerator) / Decimal(squared_volume.denominator)
            ).sqrt()
            measure = float(root / Decimal(math.factorial(degree)))
    except (InvalidOperation, OverflowError) as error:
        raise GeometryError("simplex measure is not representable") from error
    if not math.isfinite(measure) or measure <= 0.0:
        raise GeometryError("simplex measure is not representable")
    return measure


def _determinant(matrix: list[list[Fraction]]) -> Fraction:
    reduced = [row.copy() for row in matrix]
    determinant = Fraction(1)
    sign = 1

    for column in range(len(reduced)):
        pivot = next(
            (row for row in range(column, len(reduced)) if reduced[row][column] != 0),
            None,
        )
        if pivot is None:
            return Fraction()
        if pivot != column:
            reduced[column], reduced[pivot] = reduced[pivot], reduced[column]
            sign = -sign
        pivot_value = reduced[column][column]
        determinant *= pivot_value
        for row in range(column + 1, len(reduced)):
            factor = reduced[row][column] / pivot_value
            for trailing in range(column, len(reduced)):
                reduced[row][trailing] -= factor * reduced[column][trailing]

    return determinant * sign


def _scaled_measure(
    scales: FloatArray,
    diagonal: FloatArray,
    degree: int,
) -> float:
    product_mantissa = 1.0
    product_exponent = 0

    for scale, factor in zip(scales, diagonal, strict=True):
        scale_mantissa, scale_exponent = math.frexp(float(scale))
        factor_mantissa, factor_exponent = math.frexp(float(factor))
        if factor_mantissa == 0.0:
            raise GeometryError("simplex is degenerate")
        product_mantissa *= scale_mantissa * factor_mantissa
        product_mantissa, shift = math.frexp(product_mantissa)
        product_exponent += scale_exponent + factor_exponent + shift

    for divisor in range(2, degree + 1):
        product_mantissa /= divisor
        product_mantissa, shift = math.frexp(product_mantissa)
        product_exponent += shift

    try:
        measure = math.ldexp(product_mantissa, product_exponent)
    except OverflowError as error:
        raise GeometryError("simplex measure is not representable") from error
    if not math.isfinite(measure) or measure <= 0.0:
        raise GeometryError("simplex measure is not representable")
    return measure


__all__ = ["Geometry", "GeometryError"]
