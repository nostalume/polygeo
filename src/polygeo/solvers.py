"""Prepared sparse direct solving with residual certification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, final

import numpy as np
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
        if self.residual_scale == 0.0:
            if self.residual_norm != 0.0 or self.relative_residual != 0.0:
                raise NumericalError(
                    "zero residual scale requires zero residual evidence"
                )
            return
        expected = self.residual_norm / self.residual_scale
        if self.relative_residual != expected:
            raise NumericalError("relative residual does not match its norm and scale")
        if expected > np.sqrt(np.finfo(np.float64).eps):
            raise NumericalError("solve failed residual certification")


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
    residual_norm = 0.0
    product_norm = 0.0
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
        exact_residual = exact_product - binary64_lattice(float(rhs[row]))
        try:
            product = binary64_from_lattice(exact_product)
            residual = binary64_from_lattice(exact_residual)
        except OverflowError as error:
            raise NumericalError(
                "direct solve produced a non-finite residual"
            ) from error
        if not np.isfinite(product) or not np.isfinite(residual):
            raise NumericalError("direct solve produced a non-finite residual")
        if residual == 0.0 and exact_residual != 0:
            raise NumericalError("direct solve produced an unrepresentable residual")
        product_norm = max(product_norm, abs(product))
        residual_norm = max(residual_norm, abs(residual))
    return residual_norm, max(
        product_norm,
        float(np.max(np.abs(rhs), initial=0.0)),
    )


__all__ = [
    "LinearSolution",
    "NumericalError",
    "PrepareLinearSolve",
    "PreparedLinearSolve",
    "prepare_direct",
]
