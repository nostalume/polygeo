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

_FORWARD_CONDITION_LIMIT = float(np.finfo(np.float64).eps ** (-1.0 / 8.0))


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

    def primal_measures(self, degree: int) -> FloatArray:
        if degree < 0 or degree > self._complex.dimension:
            raise GeometryError("primal-measure degree is outside the complex")
        return self._measures[degree].copy()

    def dual_measures(self, degree: int) -> FloatArray:
        if degree < 0 or degree > self._complex.dimension:
            raise GeometryError("dual-measure degree is outside the complex")
        return _compute_dual_measures(
            self._complex,
            self._positions,
            self._measures,
            degree,
        )


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

    with np.errstate(all="ignore"):
        normalized = edges / scales
    try:
        _, triangular = np.linalg.qr(normalized, mode="reduced")
        with np.errstate(all="ignore"):
            condition = float(np.linalg.cond(triangular))
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
    if (
        float(np.min(diagonal)) <= suspicion
        or not math.isfinite(condition)
        or condition > _FORWARD_CONDITION_LIMIT
    ):
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


def _compute_dual_measures(
    complex_: _GeometryDomain,
    positions: FloatArray,
    primal_measures: tuple[FloatArray, ...],
    requested_degree: int,
) -> FloatArray:
    dimension = complex_.dimension
    upper = np.ones(complex_.simplex_count(dimension), dtype=np.float64)
    if requested_degree == dimension:
        return upper

    face_indices = _immediate_face_indices(complex_)
    steps = _circumcentric_steps(
        complex_,
        positions,
        primal_measures,
        face_indices,
    )

    for lower_degree in range(dimension - 1, requested_degree - 1, -1):
        upper_basis = complex_.simplices(lower_degree + 1)
        faces_per_upper = lower_degree + 2
        upper_indices = np.repeat(
            np.arange(len(upper_basis), dtype=np.int64), faces_per_upper
        )
        lower_indices = face_indices[lower_degree + 1].reshape(-1)
        divisor = dimension - lower_degree
        contributions = _scaled_products(
            steps[lower_degree + 1].reshape(-1),
            upper[upper_indices],
            divisor,
        )
        upper = _sum_incident(
            lower_indices,
            contributions,
            complex_.simplex_count(lower_degree),
        )

    return upper


def _immediate_face_indices(
    complex_: _GeometryDomain,
) -> tuple[NDArray[np.int64], ...]:
    result = [np.empty((complex_.simplex_count(0), 0), dtype=np.int64)]
    for degree in range(1, complex_.dimension + 1):
        lower_lookup = {
            tuple(int(vertex) for vertex in simplex): index
            for index, simplex in enumerate(complex_.simplices(degree - 1))
        }
        basis = complex_.simplices(degree)
        indices = np.empty((len(basis), degree + 1), dtype=np.int64)
        for upper_index, simplex in enumerate(basis):
            for omitted in range(degree + 1):
                face = tuple(
                    int(vertex)
                    for local, vertex in enumerate(simplex)
                    if local != omitted
                )
                indices[upper_index, omitted] = lower_lookup[face]
        result.append(indices)
    return tuple(result)


def _circumcentric_steps(
    complex_: _GeometryDomain,
    positions: FloatArray,
    primal_measures: tuple[FloatArray, ...],
    face_indices: tuple[NDArray[np.int64], ...],
) -> tuple[FloatArray, ...]:
    result = [np.empty((complex_.simplex_count(0), 0), dtype=np.float64)]
    for degree in range(1, complex_.dimension + 1):
        faces = face_indices[degree]
        heights = _scaled_ratios(
            np.broadcast_to(primal_measures[degree][:, None], faces.shape),
            primal_measures[degree - 1][faces],
            degree,
        )
        result.append(
            _signed_circumcentric_steps(
                positions[complex_.simplices(degree)],
                heights,
            )
        )
    return tuple(result)


def _signed_circumcentric_steps(
    points: FloatArray,
    heights: FloatArray,
) -> FloatArray:
    degree = points.shape[1] - 1
    with np.errstate(all="ignore"):
        edges = np.swapaxes(points[:, 1:] - points[:, :1], 1, 2)
    if not np.all(np.isfinite(edges)):
        raise GeometryError("circumcenter differences are not representable")

    scales = np.max(np.abs(edges), axis=1)
    if np.any(scales == 0.0) or not np.all(np.isfinite(scales)):
        raise GeometryError("simplex is degenerate")
    with np.errstate(all="ignore"):
        normalized = edges / scales[:, None, :]

    try:
        _, triangular = np.linalg.qr(normalized, mode="reduced")
        with np.errstate(all="ignore"):
            condition = np.linalg.cond(triangular)
    except np.linalg.LinAlgError:
        return _exact_circumcentric_steps(points, heights)

    diagonal = np.abs(np.diagonal(triangular, axis1=1, axis2=2))
    suspicion = (
        np.finfo(np.float64).eps
        * max(normalized.shape[1:])
        * np.max(diagonal, axis=1)
        * 64.0
    )
    fallback = (
        (np.min(diagonal, axis=1) <= suspicion)
        | ~np.isfinite(condition)
        | (condition > _FORWARD_CONDITION_LIMIT)
    )
    healthy = np.flatnonzero(~fallback)

    steps = np.empty((len(points), degree + 1), dtype=np.float64)
    if len(healthy):
        normalized_healthy = normalized[healthy]
        scales_healthy = scales[healthy]
        triangular_healthy = triangular[healthy]
        with np.errstate(all="ignore"):
            right = (
                0.5
                * scales_healthy
                * np.sum(normalized_healthy * normalized_healthy, axis=1)
            )
        try:
            affine = np.linalg.solve(
                np.swapaxes(triangular_healthy, 1, 2), right[..., None]
            )[..., 0]
            normalized_coefficients = np.linalg.solve(
                triangular_healthy, affine[..., None]
            )[..., 0]
        except np.linalg.LinAlgError:
            fallback[healthy] = True
        else:
            with np.errstate(all="ignore"):
                coefficients = normalized_coefficients / scales_healthy
                barycentric = np.concatenate(
                    (
                        1.0 - np.sum(coefficients, axis=1, keepdims=True),
                        coefficients,
                    ),
                    axis=1,
                )
                solved = np.einsum("sji,sj->si", triangular_healthy, affine)
                residual = solved - right

            finite = np.all(np.isfinite(barycentric), axis=1)
            with np.errstate(all="ignore"):
                residual_scale = np.maximum(
                    np.max(np.abs(right), axis=1),
                    np.max(np.abs(solved), axis=1),
                )
                residual_ok = np.max(np.abs(residual), axis=1) <= (
                    np.finfo(np.float64).eps * max(1, degree) * residual_scale * 64.0
                )
                sign_suspicion = (
                    np.finfo(np.float64).eps
                    * np.maximum(1.0, np.sum(np.abs(barycentric), axis=1))
                    * 64.0
                )
                coefficients_clear = np.all(
                    np.abs(barycentric) > sign_suspicion[:, None], axis=1
                )
            accepted = finite & residual_ok & coefficients_clear
            accepted_indices = healthy[accepted]
            accepted_steps = _scaled_products(
                barycentric[accepted].reshape(-1),
                heights[accepted_indices].reshape(-1),
                1,
            )
            steps[accepted_indices] = accepted_steps.reshape(-1, degree + 1)
            fallback[healthy[~accepted]] = True

    for index in np.flatnonzero(fallback):
        barycentric = _exact_barycentric(points[index])
        for local, coefficient in enumerate(barycentric):
            steps[index, local] = _exact_scaled_value(
                coefficient,
                float(heights[index, local]),
            )
    return steps


def _exact_circumcentric_steps(
    points: FloatArray,
    heights: FloatArray,
) -> FloatArray:
    steps = np.empty((len(points), points.shape[1]), dtype=np.float64)
    for index, simplex_points in enumerate(points):
        for local, coefficient in enumerate(_exact_barycentric(simplex_points)):
            steps[index, local] = _exact_scaled_value(
                coefficient,
                float(heights[index, local]),
            )
    return steps


def _exact_barycentric(points: FloatArray) -> list[Fraction]:
    degree = len(points) - 1
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
    right = [gram[index][index] / 2 for index in range(degree)]
    coefficients = _solve_exact(gram, right)
    return [1 - sum(coefficients, start=Fraction()), *coefficients]


def _exact_scaled_value(coefficient: Fraction, scale: float) -> float:
    exact = coefficient * Fraction(scale)
    try:
        value = float(exact)
    except OverflowError as error:
        raise GeometryError("dual step is not representable") from error
    if not math.isfinite(value) or (value == 0.0 and exact != 0):
        raise GeometryError("dual step is not representable")
    return value


def _solve_exact(
    matrix: list[list[Fraction]],
    right: list[Fraction],
) -> list[Fraction]:
    augmented = [[*row, value] for row, value in zip(matrix, right, strict=True)]
    size = len(augmented)
    for column in range(size):
        pivot = next(
            (row for row in range(column, size) if augmented[row][column] != 0),
            None,
        )
        if pivot is None:
            raise GeometryError("simplex is degenerate")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(
                    augmented[row], augmented[column], strict=True
                )
            ]
    return [row[-1] for row in augmented]


def _scaled_ratios(
    numerator: FloatArray,
    denominator: FloatArray,
    multiplier: int,
) -> FloatArray:
    with np.errstate(all="ignore"):
        numerator_mantissa, numerator_exponent = np.frexp(numerator)
        denominator_mantissa, denominator_exponent = np.frexp(denominator)
        values = np.ldexp(
            multiplier * numerator_mantissa / denominator_mantissa,
            numerator_exponent - denominator_exponent,
        )
    if not np.all(np.isfinite(values)) or np.any(values == 0.0):
        raise GeometryError("simplex altitude is not representable")
    return values


def _scaled_products(
    left: FloatArray,
    right: FloatArray,
    divisor: int,
) -> FloatArray:
    left_mantissa, left_exponent = np.frexp(left)
    right_mantissa, right_exponent = np.frexp(right)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        values = np.ldexp(
            left_mantissa * right_mantissa / divisor,
            left_exponent + right_exponent,
        )
    nonzero = (left != 0.0) & (right != 0.0)
    if not np.all(np.isfinite(values)) or np.any(nonzero & (values == 0.0)):
        raise GeometryError("dual contribution is not representable")
    return values


def _sum_incident(
    lower_indices: NDArray[np.int64],
    contributions: FloatArray,
    lower_count: int,
) -> FloatArray:
    values = np.zeros(lower_count, dtype=np.float64)
    order = np.argsort(lower_indices, kind="stable")
    rows = lower_indices[order]
    terms = contributions[order]
    boundaries = np.flatnonzero(np.concatenate(([True], rows[1:] != rows[:-1], [True])))
    try:
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            values[rows[start]] = math.fsum(terms[start:stop])
    except OverflowError as error:
        raise GeometryError("dual measure is not representable") from error
    if not np.all(np.isfinite(values)):
        raise GeometryError("dual measure is not representable")
    return values


__all__ = ["Geometry", "GeometryError"]
