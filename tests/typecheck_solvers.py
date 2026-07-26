from typing import Literal, assert_type

import numpy as np
from scipy.sparse import csr_array

from polygeo import (
    ORDINARY_FORM,
    AssembledSystem,
    BoundaryRegular,
    BoundaryUnknown,
    CochainSpace,
    CochainSubspace,
    Complex,
    ConnectivityUnknown,
    DirichletProblem,
    FieldSemantics,
    Form,
    LinearMap,
    LinearSolution,
    OrientationUnknown,
    OrdinaryForm,
    PrepareLinearSolve,
    PreparedLinearSolve,
    eliminate_dirichlet,
    prepare_direct,
)


type Regular = Complex[
    BoundaryUnknown,
    OrientationUnknown,
    ConnectivityUnknown,
    BoundaryRegular,
]
type Parent = CochainSpace[Regular, Literal[0]]
type Reduced = CochainSubspace[Parent]


class AlternateSemantics(FieldSemantics):
    pass


raw = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
regular: Regular = raw.boundary_regular()
parent: Parent = regular.cochain_space(0)
operator = LinearMap(parent, parent, csr_array(np.eye(parent.size)))
rhs = parent.form(np.ones(parent.size), AlternateSemantics())
system = AssembledSystem(operator, rhs)

preparer: PrepareLinearSolve[Parent, Parent] = prepare_direct
prepared: PreparedLinearSolve[Parent, Parent] = prepare_direct(operator)
solution = prepared(rhs)
assert_type(solution, LinearSolution[Parent, Parent, AlternateSemantics])
assert_type(solution.form, Form[Parent, AlternateSemantics])
assert_type(solution.equation_space, Parent)
assert_type(
    system.solve(prepare_direct), LinearSolution[Parent, Parent, AlternateSemantics]
)

region = CochainSubspace(parent, np.array([0], dtype=np.int64))
problem: DirichletProblem[Parent, AlternateSemantics] = eliminate_dirichlet(
    system,
    region,
    region.form(np.zeros(1), AlternateSemantics()),
)
assert_type(
    problem.solve(prepare_direct),
    LinearSolution[Parent, Reduced, AlternateSemantics],
)
assert_type(problem.solve(prepare_direct).equation_space, Reduced)

ordinary = parent.form(np.ones(parent.size), ORDINARY_FORM)
assert_type(prepared(ordinary), LinearSolution[Parent, Parent, OrdinaryForm])
