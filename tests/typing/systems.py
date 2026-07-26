from typing import Literal, assert_type

import numpy as np
from scipy.sparse import csr_array

from polygeo import (
    AssembledSystem,
    CodimensionOneRegular,
    BoundaryUnknown,
    CochainSpace,
    CochainSubspace,
    Complex,
    ConnectivityUnknown,
    DirichletProblem,
    FieldSemantics,
    Form,
    LinearMap,
    OrientationUnknown,
    eliminate_dirichlet,
)


type Regular = Complex[
    BoundaryUnknown,
    OrientationUnknown,
    ConnectivityUnknown,
    CodimensionOneRegular,
]
type Parent = CochainSpace[Regular, Literal[0]]
type Reduced = CochainSubspace[Parent]


class AlternateSemantics(FieldSemantics):
    pass


raw = Complex.from_maximal_simplices(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))
regular: Regular = raw.codimension_one_regular()
parent = regular.cochain_space(0)
operator = LinearMap(parent, parent, csr_array(np.eye(parent.size)))
rhs = parent.form(np.ones(parent.size), AlternateSemantics())
system = AssembledSystem(operator, rhs)
assert_type(system, AssembledSystem[Parent, Parent, AlternateSemantics])
assert_type(system.operator, LinearMap[Parent, Parent])
assert_type(system.rhs, Form[Parent, AlternateSemantics])

region = CochainSubspace(parent, np.array([0], dtype=np.int64))
values = region.form(np.zeros(1), AlternateSemantics())
problem = eliminate_dirichlet(system, region, values)
assert_type(problem, DirichletProblem[Parent, AlternateSemantics])
assert_type(problem.operator, LinearMap[Reduced, Reduced])
assert_type(problem.rhs, Form[Reduced, AlternateSemantics])
assert_type(problem.parent, Parent)
assert_type(problem.boundary, Reduced)
assert_type(problem.interior, Reduced)

interior = problem.interior.form(np.zeros(problem.interior.size), AlternateSemantics())
assert_type(problem.reconstruct(interior), Form[Parent, AlternateSemantics])
