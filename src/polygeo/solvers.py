"""Prepared sparse direct solving with residual certification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, final

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.sparse import csr_array
from scipy.sparse.linalg import SuperLU, splu

from .numerics import (
    BINARY64_PRODUCT_LATTICE_BITS,
    binary64_from_lattice,
    binary64_lattice,
    binary64_ratio,
)
from .operators import LinearMap
from .simplicial import FieldSemantics, Form, _CoefficientSpace

_SQRT_EPS64 = float(np.sqrt(np.finfo(np.float64).eps))


class NumericalError(ValueError):
    """Invalid system, failed factorization, or uncertified solve."""


@dataclass(frozen=True, slots=True)
@final
class LinearSolution[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
    Semantics: FieldSemantics,
]:
    """A form with consistent residual evidence in its equation space."""

    form: Form[UnknownSpace, Semantics]
    equation_space: EquationSpace
    residual_norm: float
    residual_scale: float
    relative_residual: float

    def __post_init__(self) -> None:
        evidence = (
            self.residual_norm,
            self.residual_scale,
            self.relative_residual,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in evidence):
            raise NumericalError("residual evidence must be finite and nonnegative")
        zero_scale = self.residual_scale == 0.0
        if zero_scale and (self.residual_norm or self.relative_residual):
            raise NumericalError("zero residual scale requires zero residual evidence")
        expected = 0.0 if zero_scale else self.residual_norm / self.residual_scale
        if self.relative_residual != expected:
            raise NumericalError("relative residual does not match its norm and scale")
        if expected > _SQRT_EPS64:
            raise NumericalError("solve failed residual certification")


@dataclass(frozen=True, slots=True)
@final
class LeastSquaresSolution[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
    Semantics: FieldSemantics,
]:
    """A least-squares form with residual and condition-admission evidence."""

    form: Form[UnknownSpace, Semantics]
    equation_space: EquationSpace
    normal_residual_norm: float
    normal_residual_scale: float
    relative_normal_residual: float
    condition_indicator: float
    condition_limit: float

    def __post_init__(self) -> None:
        residual = (
            self.normal_residual_norm,
            self.normal_residual_scale,
            self.relative_normal_residual,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in residual):
            raise NumericalError(
                "normal-residual evidence must be finite and nonnegative"
            )
        zero_scale = self.normal_residual_scale == 0.0
        if zero_scale and (self.normal_residual_norm or self.relative_normal_residual):
            raise NumericalError("zero normal-residual scale requires zero residual")
        expected = (
            0.0
            if zero_scale
            else self.normal_residual_norm / self.normal_residual_scale
        )
        if self.relative_normal_residual != expected:
            raise NumericalError("relative normal residual is inconsistent")
        if self.relative_normal_residual > _SQRT_EPS64:
            raise NumericalError("least-squares normal residual is uncertified")
        if not np.isfinite(self.condition_limit) or self.condition_limit <= 0.0:
            raise NumericalError("condition limit must be finite and positive")
        if (
            not np.isfinite(self.condition_indicator)
            or self.condition_indicator < 0.0
            or self.condition_indicator > self.condition_limit
        ):
            raise NumericalError("least-squares condition indicator is not admitted")


class PreparedLeastSquares[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
](Protocol):
    """Prepared rectangular solve reusable across right-hand sides."""

    def __call__[Semantics: FieldSemantics](
        self,
        rhs: Form[EquationSpace, Semantics],
    ) -> LeastSquaresSolution[UnknownSpace, EquationSpace, Semantics]: ...


class PrepareLeastSquares[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
](Protocol):
    """Injected behavior that prepares one full-column-rank map."""

    def __call__(
        self,
        operator: LinearMap[UnknownSpace, EquationSpace],
    ) -> PreparedLeastSquares[UnknownSpace, EquationSpace]: ...


class _QRSolve[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
]:
    __slots__ = (
        "_condition_indicator",
        "_matrix",
        "_permutation",
        "_q",
        "_r",
        "_scales",
        "_source",
        "_target",
    )

    def __init__(
        self,
        operator: LinearMap[UnknownSpace, EquationSpace],
        matrix: np.ndarray,
        q: np.ndarray,
        r: np.ndarray,
        permutation: np.ndarray,
        scales: np.ndarray,
        condition_indicator: float,
    ) -> None:
        self._source = operator.source
        self._target = operator.target
        self._matrix = matrix
        self._q = q
        self._r = r
        self._permutation = permutation
        self._scales = scales
        self._condition_indicator = condition_indicator

    def __call__[Semantics: FieldSemantics](
        self,
        rhs: Form[EquationSpace, Semantics],
    ) -> LeastSquaresSolution[UnknownSpace, EquationSpace, Semantics]:
        if not rhs.space.same_space(self._target):
            raise NumericalError(
                "right-hand side does not belong to the equation space"
            )
        coefficients = rhs.coefficients()
        try:
            projected = self._q.T @ coefficients
            permuted = solve_triangular(self._r, projected, check_finite=False)
            normalized = np.empty(self._source.size, dtype=np.float64)
            normalized[self._permutation] = permuted
            solved = normalized / self._scales
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            raise NumericalError("least-squares solve failed") from error
        if not np.all(np.isfinite(solved)):
            raise NumericalError("least-squares solve produced non-finite coefficients")
        normal_norm, normal_scale = _normal_residual_evidence(
            self._matrix, solved, coefficients
        )
        relative = 0.0 if normal_scale == 0.0 else normal_norm / normal_scale
        return LeastSquaresSolution(
            Form(self._source, solved, rhs.semantics),
            self._target,
            normal_norm,
            normal_scale,
            relative,
            self._condition_indicator,
            _SQRT_EPS64,
        )


class _EmptyLeastSquares[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
]:
    __slots__ = ("_source", "_target")

    def __init__(self, operator: LinearMap[UnknownSpace, EquationSpace]) -> None:
        self._source = operator.source
        self._target = operator.target

    def __call__[Semantics: FieldSemantics](
        self,
        rhs: Form[EquationSpace, Semantics],
    ) -> LeastSquaresSolution[UnknownSpace, EquationSpace, Semantics]:
        if not rhs.space.same_space(self._target):
            raise NumericalError(
                "right-hand side does not belong to the equation space"
            )
        return LeastSquaresSolution(
            Form(self._source, np.empty(0, dtype=np.float64), rhs.semantics),
            self._target,
            0.0,
            0.0,
            0.0,
            0.0,
            _SQRT_EPS64,
        )


def prepare_least_squares[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
](
    operator: LinearMap[UnknownSpace, EquationSpace],
) -> PreparedLeastSquares[UnknownSpace, EquationSpace]:
    """Prepare a condition-gated, column-scaled pivoted QR solve."""
    rows, columns = operator._matrix.shape
    if rows < columns:
        raise NumericalError(
            "least-squares map requires at least as many rows as columns"
        )
    if columns == 0:
        return _EmptyLeastSquares(operator)
    matrix = operator._matrix.toarray().astype(np.float64, copy=False)
    normalized = np.empty_like(matrix)
    scales = np.empty(columns, dtype=np.float64)
    for column in range(columns):
        values = matrix[:, column]
        magnitude = float(np.max(np.abs(values)))
        if not np.isfinite(magnitude) or magnitude == 0.0:
            raise NumericalError("least-squares map is rank deficient")
        with np.errstate(all="ignore"):
            scale = magnitude * float(np.linalg.norm(values / magnitude))
        if not np.isfinite(scale) or scale == 0.0:
            raise NumericalError("least-squares column scale is not representable")
        scales[column] = scale
        normalized[:, column] = values / scale
    try:
        q, r, permutation = qr(
            normalized, mode="economic", pivoting=True, check_finite=False
        )
        diagonal = np.abs(np.diag(r))
        threshold = max(rows, columns) * np.finfo(np.float64).eps * float(diagonal[0])
        if np.any(diagonal <= threshold):
            raise NumericalError("least-squares map is rank deficient")
        condition = float(np.linalg.cond(r))
    except NumericalError:
        raise
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        raise NumericalError("least-squares preparation failed") from error
    dimension = rows + columns
    epsilon = np.finfo(np.float64).eps
    gamma = dimension * epsilon / (1.0 - dimension * epsilon)
    condition_indicator = condition * gamma
    limit = _SQRT_EPS64
    if not np.isfinite(condition_indicator) or condition_indicator > limit:
        raise NumericalError(
            "least-squares condition indicator exceeds its admitted limit"
        )
    return _QRSolve(operator, matrix, q, r, permutation, scales, condition_indicator)


def _normal_residual_evidence(
    matrix: np.ndarray, solved: np.ndarray, rhs: np.ndarray
) -> tuple[float, float]:
    matrix_scale = float(np.max(np.abs(matrix), initial=0.0))
    solution_scale = float(np.max(np.abs(solved), initial=0.0))
    rhs_scale = float(np.max(np.abs(rhs), initial=0.0))
    product_scale = Fraction(matrix_scale) * Fraction(solution_scale)
    rhs_scale_exact = Fraction(rhs_scale)
    common_scale = max(product_scale, rhs_scale_exact)
    if common_scale == 0:
        return 0.0, 0.0
    product_ratio = float(product_scale / common_scale)
    rhs_ratio = float(rhs_scale_exact / common_scale)
    if (product_scale != 0 and product_ratio == 0.0) or (
        rhs_scale_exact != 0 and rhs_ratio == 0.0
    ):
        raise NumericalError("least-squares evidence scaling underflowed")
    with np.errstate(all="ignore"):
        normalized_matrix = matrix / matrix_scale
        normalized_solution = (
            solved / solution_scale if solution_scale else np.zeros_like(solved)
        )
        normalized_rhs = rhs / rhs_scale if rhs_scale else np.zeros_like(rhs)
        product = product_ratio * (normalized_matrix @ normalized_solution)
        residual = product - rhs_ratio * normalized_rhs
        normal = normalized_matrix.T @ residual
        absolute = np.abs(normalized_matrix)
        matrix_norm = float(np.max(np.sum(absolute, axis=1), initial=0.0))
        transpose_norm = float(np.max(np.sum(absolute, axis=0), initial=0.0))
        scale = transpose_norm * (product_ratio * matrix_norm + rhs_ratio)
    evidence = (normal, product, residual, scale)
    if any(not np.all(np.isfinite(value)) for value in evidence):
        raise NumericalError("least-squares normal residual is not representable")
    norm = float(np.max(np.abs(normal), initial=0.0))
    if scale == 0.0 and norm != 0.0:
        raise NumericalError("zero normal-residual scale has nonzero residual")
    return norm, scale


class PreparedLinearSolve[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
](Protocol):
    """Operation-scoped numerical behavior reusable across right-hand sides."""

    def __call__[Semantics: FieldSemantics](
        self,
        rhs: Form[EquationSpace, Semantics],
    ) -> LinearSolution[UnknownSpace, EquationSpace, Semantics]: ...


class PrepareLinearSolve[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
](Protocol):
    """Injected behavior that prepares one linear operator."""

    def __call__(
        self,
        operator: LinearMap[UnknownSpace, EquationSpace],
    ) -> PreparedLinearSolve[UnknownSpace, EquationSpace]: ...


class _DirectSolve[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
]:
    __slots__ = ("_coefficient_parts", "_factor", "_rows", "_source", "_target")

    def __init__(
        self,
        operator: LinearMap[UnknownSpace, EquationSpace],
        rows: csr_array,
        coefficient_parts: tuple[tuple[int, int], ...],
        factor: SuperLU,
    ) -> None:
        self._source = operator.source
        self._target = operator.target
        self._rows = rows
        self._coefficient_parts = coefficient_parts
        self._factor = factor

    def __call__[Semantics: FieldSemantics](
        self,
        rhs: Form[EquationSpace, Semantics],
    ) -> LinearSolution[UnknownSpace, EquationSpace, Semantics]:
        if not rhs.space.same_space(self._target):
            raise NumericalError(
                "right-hand side does not belong to the equation space"
            )
        coefficients = rhs.coefficients()
        try:
            solved = np.asarray(self._factor.solve(coefficients), dtype=np.float64)
        except (RuntimeError, ValueError) as error:
            raise NumericalError("direct solve failed") from error
        if solved.shape != (self._source.size,) or not np.all(np.isfinite(solved)):
            raise NumericalError("direct solve produced non-finite coefficients")

        residual_norm, residual_scale = _residual_evidence(
            self._rows,
            self._coefficient_parts,
            solved,
            coefficients,
        )
        relative_residual = (
            0.0
            if residual_scale == 0.0 and residual_norm == 0.0
            else residual_norm / residual_scale
        )
        form = Form(self._source, solved, rhs.semantics)
        return LinearSolution(
            form,
            self._target,
            residual_norm,
            residual_scale,
            relative_residual,
        )


def prepare_direct[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
](
    operator: LinearMap[UnknownSpace, EquationSpace],
) -> PreparedLinearSolve[UnknownSpace, EquationSpace]:
    """Factor a finite square sparse operator for repeated direct solves."""
    if operator.source.size != operator.target.size:
        raise NumericalError("direct factorization requires a square operator")
    rows = operator._matrix
    matrix = rows.tocsc(copy=False)
    if not np.all(np.isfinite(matrix.data)):
        raise NumericalError("direct factorization requires finite coefficients")
    try:
        factor = splu(matrix)
    except (RuntimeError, ValueError) as error:
        raise NumericalError("direct factorization failed") from error
    coefficient_parts = tuple(binary64_ratio(float(value)) for value in rows.data)
    return _DirectSolve(operator, rows, coefficient_parts, factor)


def _residual_evidence(
    matrix: csr_array,
    coefficient_parts: tuple[tuple[int, int], ...],
    solved: np.ndarray,
    rhs: np.ndarray,
) -> tuple[float, float]:
    exact_residual_norm, exact_product_norm, exact_rhs_norm = _residual_lattice_maxima(
        matrix, coefficient_parts, solved, rhs
    )
    try:
        residual_norm = binary64_from_lattice(exact_residual_norm)
        product_norm = binary64_from_lattice(exact_product_norm)
        rhs_norm = binary64_from_lattice(exact_rhs_norm)
    except OverflowError as error:
        raise NumericalError("direct solve produced a non-finite residual") from error
    if not all(np.isfinite(value) for value in (residual_norm, product_norm, rhs_norm)):
        raise NumericalError("direct solve produced a non-finite residual")
    if residual_norm == 0.0 and exact_residual_norm != 0:
        raise NumericalError("direct solve produced an unrepresentable residual")
    return residual_norm, max(product_norm, rhs_norm)


def _normalized_residual_evidence(
    matrix: csr_array,
    coefficient_parts: tuple[tuple[int, int], ...],
    solved: np.ndarray,
    rhs: np.ndarray,
) -> tuple[float, float]:
    exact_residual_norm, exact_product_norm, exact_rhs_norm = _residual_lattice_maxima(
        matrix, coefficient_parts, solved, rhs
    )
    exact_scale = max(exact_product_norm, exact_rhs_norm)
    if exact_scale == 0:
        if exact_residual_norm != 0:
            raise NumericalError("zero residual scale has a nonzero residual")
        return 0.0, 0.0
    exact_relative = Fraction(exact_residual_norm, exact_scale)
    relative = float(exact_relative)
    if relative == 0.0 and exact_residual_norm != 0:
        relative = math.nextafter(0.0, math.inf)
    elif Fraction(relative) > exact_relative:
        relative = math.nextafter(relative, 0.0)
    return relative, 1.0


def _residual_lattice_maxima(
    matrix: csr_array,
    coefficient_parts: tuple[tuple[int, int], ...],
    solved: np.ndarray,
    rhs: np.ndarray,
) -> tuple[int, int, int]:
    residual_norm = 0
    product_norm = 0
    rhs_norm = 0
    solved_parts = tuple(binary64_ratio(float(value)) for value in solved)
    for row in range(matrix.shape[0]):
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        exact_product = 0
        for offset in range(start, stop):
            coefficient_numerator, coefficient_denominator_bits = coefficient_parts[
                offset
            ]
            value_numerator, value_denominator_bits = solved_parts[
                matrix.indices[offset]
            ]
            exact_product += (coefficient_numerator * value_numerator) << (
                BINARY64_PRODUCT_LATTICE_BITS
                - coefficient_denominator_bits
                - value_denominator_bits
            )
        exact_rhs = binary64_lattice(float(rhs[row]))
        exact_residual = exact_product - exact_rhs
        product_norm = max(product_norm, abs(exact_product))
        residual_norm = max(residual_norm, abs(exact_residual))
        rhs_norm = max(rhs_norm, abs(exact_rhs))
    return residual_norm, product_norm, rhs_norm


def _linear_residual_evidence[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
    Semantics: FieldSemantics,
](
    operator: LinearMap[UnknownSpace, EquationSpace],
    solved: Form[UnknownSpace, Semantics],
    rhs: Form[EquationSpace, Semantics],
) -> tuple[float, float, float]:
    """Recompute exact binary64 residual evidence for an injected solve result."""
    if not solved.space.same_space(operator.source) or not rhs.space.same_space(
        operator.target
    ):
        raise NumericalError("residual certification spaces do not match the operator")
    matrix = operator.matrix()
    coefficient_parts = tuple(binary64_ratio(float(value)) for value in matrix.data)
    norm, scale = _residual_evidence(
        matrix,
        coefficient_parts,
        solved.coefficients(),
        rhs.coefficients(),
    )
    relative = 0.0 if scale == 0.0 else norm / scale
    if relative > _SQRT_EPS64:
        raise NumericalError("solve failed residual certification")
    return norm, scale, relative


__all__ = [
    "LeastSquaresSolution",
    "LinearSolution",
    "NumericalError",
    "PrepareLeastSquares",
    "PrepareLinearSolve",
    "PreparedLeastSquares",
    "PreparedLinearSolve",
    "prepare_direct",
    "prepare_least_squares",
]
