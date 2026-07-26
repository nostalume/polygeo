"""Typed assembled systems and essential boundary elimination."""

from __future__ import annotations

from typing import Self

import numpy as np
from scipy.sparse import csr_array

from .numerics import (
    binary64_from_lattice,
    binary64_lattice,
    binary64_sum_product_lattice,
)
from .operators import LinearMap
from .solvers import LinearSolution, PrepareLinearSolve
from .simplicial import (
    BoundaryRegular,
    BoundaryState,
    CochainSpace,
    CochainSubspace,
    Complex,
    ConnectivityState,
    FieldSemantics,
    Form,
    OrientationState,
    _CochainParent,
    _CoefficientSpace,
    topological_boundary,
)


class SystemError(ValueError):
    """Invalid assembled system, boundary elimination, or reconstruction."""


class AssembledSystem[
    UnknownSpace: _CoefficientSpace,
    EquationSpace: _CoefficientSpace,
    Semantics: FieldSemantics,
]:
    """A linear operator and right-hand side with exact coefficient spaces."""

    __slots__ = ("_operator", "_rhs")

    def __init__(
        self,
        operator: LinearMap[UnknownSpace, EquationSpace],
        rhs: Form[EquationSpace, Semantics],
    ) -> None:
        if not operator.target.same_space(rhs.space):
            raise SystemError("right-hand side does not belong to the equation space")
        self._operator = operator
        self._rhs = rhs

    @property
    def operator(self) -> LinearMap[UnknownSpace, EquationSpace]:
        return self._operator

    @property
    def rhs(self) -> Form[EquationSpace, Semantics]:
        return self._rhs

    def solve(
        self,
        prepare: PrepareLinearSolve[UnknownSpace, EquationSpace],
    ) -> LinearSolution[UnknownSpace, EquationSpace, Semantics]:
        return prepare(self._operator)(self._rhs)


class DirichletProblem[
    ParentSpace: _CochainParent,
    Semantics: FieldSemantics,
]:
    """A reduced essential-boundary problem with exact reconstruction data."""

    __slots__ = (
        "_boundary",
        "_interior",
        "_operator",
        "_parent",
        "_rhs",
        "_values",
    )
    _boundary: CochainSubspace[ParentSpace]
    _interior: CochainSubspace[ParentSpace]
    _operator: LinearMap[
        CochainSubspace[ParentSpace],
        CochainSubspace[ParentSpace],
    ]
    _parent: ParentSpace
    _rhs: Form[CochainSubspace[ParentSpace], Semantics]
    _values: Form[CochainSubspace[ParentSpace], Semantics]

    def __init__(self) -> None:
        raise SystemError("DirichletProblem must be created by eliminate_dirichlet()")

    @classmethod
    def _admitted(
        cls,
        parent: ParentSpace,
        boundary: CochainSubspace[ParentSpace],
        interior: CochainSubspace[ParentSpace],
        operator: LinearMap[
            CochainSubspace[ParentSpace],
            CochainSubspace[ParentSpace],
        ],
        rhs: Form[CochainSubspace[ParentSpace], Semantics],
        values: Form[CochainSubspace[ParentSpace], Semantics],
    ) -> Self:
        if not boundary.belongs_to(parent) or not interior.belongs_to(parent):
            raise SystemError("Dirichlet spaces must use the exact parent")
        if not boundary.complement().same_space(interior):
            raise SystemError(
                "Dirichlet boundary and interior must partition the parent"
            )
        if not operator.source.same_space(interior) or not operator.target.same_space(
            interior
        ):
            raise SystemError("reduced operator does not use the interior space")
        if not rhs.space.same_space(interior):
            raise SystemError("reduced right-hand side does not use the interior space")
        if not values.space.same_space(boundary):
            raise SystemError("boundary values do not use the boundary space")
        if not rhs.uses_semantics(values.semantics):
            raise SystemError("Dirichlet data use different field semantics")
        admitted = object.__new__(cls)
        admitted._parent = parent
        admitted._boundary = boundary
        admitted._interior = interior
        admitted._operator = operator
        admitted._rhs = rhs
        admitted._values = values
        return admitted

    @property
    def parent(self) -> ParentSpace:
        return self._parent

    @property
    def boundary(self) -> CochainSubspace[ParentSpace]:
        return self._boundary

    @property
    def interior(self) -> CochainSubspace[ParentSpace]:
        return self._interior

    @property
    def operator(
        self,
    ) -> LinearMap[
        CochainSubspace[ParentSpace],
        CochainSubspace[ParentSpace],
    ]:
        return self._operator

    @property
    def rhs(self) -> Form[CochainSubspace[ParentSpace], Semantics]:
        return self._rhs

    def reconstruct(
        self,
        interior: Form[CochainSubspace[ParentSpace], Semantics],
    ) -> Form[ParentSpace, Semantics]:
        if not interior.space.same_space(self._interior):
            raise SystemError(
                "reconstruction form does not belong to the interior space"
            )
        if not self._rhs.uses_semantics(interior.semantics):
            raise SystemError("reconstruction form uses different field semantics")
        coefficients = np.zeros(self._parent.size, dtype=np.float64)
        coefficients[self._boundary.indices()] = self._values.coefficients()
        coefficients[self._interior.indices()] = interior.coefficients()
        return Form(self._parent, coefficients, interior.semantics)

    def solve(
        self,
        prepare: PrepareLinearSolve[
            CochainSubspace[ParentSpace],
            CochainSubspace[ParentSpace],
        ],
    ) -> LinearSolution[
        ParentSpace,
        CochainSubspace[ParentSpace],
        Semantics,
    ]:
        reduced = prepare(self._operator)(self._rhs)
        return LinearSolution(
            self.reconstruct(reduced.form),
            reduced.equation_space,
            reduced.residual_norm,
            reduced.residual_scale,
            reduced.relative_residual,
        )


def eliminate_dirichlet[
    B: BoundaryState,
    O: OrientationState,
    C: ConnectivityState,
    T: BoundaryRegular,
    Degree: int,
    Semantics: FieldSemantics,
](
    system: AssembledSystem[
        CochainSpace[Complex[B, O, C, T], Degree],
        CochainSpace[Complex[B, O, C, T], Degree],
        Semantics,
    ],
    region: CochainSubspace[CochainSpace[Complex[B, O, C, T], Degree]],
    values: Form[
        CochainSubspace[CochainSpace[Complex[B, O, C, T], Degree]],
        Semantics,
    ],
) -> DirichletProblem[
    CochainSpace[Complex[B, O, C, T], Degree],
    Semantics,
]:
    """Eliminate prescribed unknowns lying on the true topological boundary."""
    parent = system.operator.source
    if not system.operator.target.same_space(parent):
        raise SystemError("Dirichlet elimination requires an endomorphism")
    if not region.belongs_to(parent):
        raise SystemError("Dirichlet region must use the exact unknown space")
    if not values.space.same_space(region):
        raise SystemError("boundary values do not belong to the boundary value space")
    if not system.rhs.uses_semantics(values.semantics):
        raise SystemError("Dirichlet data use different field semantics")

    canonical_boundary = topological_boundary(parent.complex)
    boundary_mask = canonical_boundary.mask(parent.degree)
    boundary_indices = region.indices()
    if np.any(~boundary_mask[boundary_indices]):
        raise SystemError("Dirichlet region must lie on the topological boundary")

    interior = region.complement()
    interior_indices = interior.indices()
    matrix = system.operator.matrix()
    reduced_matrix = matrix[interior_indices][:, interior_indices].tocsr()
    coupling = matrix[interior_indices][:, boundary_indices].tocsr()
    reduced_coefficients = _reduce_rhs(
        system.rhs.coefficients()[interior_indices],
        coupling,
        values.coefficients(),
    )

    reduced_operator = LinearMap(interior, interior, reduced_matrix)
    reduced_rhs = Form(interior, reduced_coefficients, system.rhs.semantics)
    return DirichletProblem._admitted(
        parent,
        region,
        interior,
        reduced_operator,
        reduced_rhs,
        values,
    )


def _reduce_rhs(
    rhs: np.ndarray,
    coupling: csr_array,
    boundary_values: np.ndarray,
) -> np.ndarray:
    reduced = np.empty_like(rhs)
    for row in range(len(rhs)):
        reduced[row] = _exact_reduced_row(
            float(rhs[row]),
            coupling,
            row,
            boundary_values,
        )
    return reduced


def _exact_reduced_row(
    rhs: float,
    coupling: csr_array,
    row: int,
    boundary_values: np.ndarray,
) -> float:
    start = int(coupling.indptr[row])
    stop = int(coupling.indptr[row + 1])
    if start == stop:
        return rhs
    exact = binary64_lattice(rhs) - binary64_sum_product_lattice(
        (float(value) for value in coupling.data[start:stop]),
        (
            float(boundary_values[coupling.indices[offset]])
            for offset in range(start, stop)
        ),
    )
    try:
        reduced = binary64_from_lattice(exact)
    except OverflowError as error:
        raise SystemError(
            "Dirichlet elimination produced an unrepresentable right-hand side"
        ) from error
    if not np.isfinite(reduced) or (reduced == 0.0 and exact != 0):
        raise SystemError(
            "Dirichlet elimination produced an unrepresentable right-hand side"
        )
    return reduced


__all__ = [
    "AssembledSystem",
    "DirichletProblem",
    "SystemError",
    "eliminate_dirichlet",
]
