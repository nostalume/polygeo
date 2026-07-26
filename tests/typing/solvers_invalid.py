# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, subclass-of-final-class

import numpy as np
from scipy.sparse import csr_array

from polygeo import (
    ORDINARY_FORM,
    AssembledSystem,
    CochainSubspace,
    Complex,
    LinearMap,
    LinearSolution,
    eliminate_dirichlet,
    prepare_direct,
)


domain = Complex.from_maximal_simplices(
    np.array([[0, 1, 2]], dtype=np.int64)
).codimension_one_regular()
C0 = domain.cochain_space(0)
C1 = domain.cochain_space(1)
identity0 = LinearMap(C0, C0, csr_array(np.eye(C0.size)))
identity1 = LinearMap(C1, C1, csr_array(np.eye(C1.size)))
rhs0 = C0.form(np.ones(C0.size), ORDINARY_FORM)
rhs1 = C1.form(np.ones(C1.size), ORDINARY_FORM)
prepared0 = prepare_direct(identity0)
prepared1 = prepare_direct(identity1)

prepared0(rhs1)
AssembledSystem(identity0, rhs0).solve(prepared0)
AssembledSystem(identity0, rhs0).solve(prepared1)

region0 = CochainSubspace(C0, np.array([0], dtype=np.int64))
problem0 = eliminate_dirichlet(
    AssembledSystem(identity0, rhs0),
    region0,
    region0.form(np.zeros(1), ORDINARY_FORM),
)
problem0.solve(prepared1)


class ForgedSolution(LinearSolution):
    pass
