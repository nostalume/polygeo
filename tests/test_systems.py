from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_array

from polygeo import (
    ORDINARY_FORM,
    AssembledSystem,
    CochainSubspace,
    Complex,
    DirichletProblem,
    FieldSemantics,
    LinearMap,
    SystemError,
    eliminate_dirichlet,
)


def _disk() -> Complex:
    return Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    ).codimension_one_regular()


def _cycle_operator(parent) -> LinearMap:
    matrix = csr_array(
        np.array(
            [
                [2.0, -1.0, 0.0, -1.0],
                [-1.0, 2.0, -1.0, 0.0],
                [0.0, -1.0, 2.0, -1.0],
                [-1.0, 0.0, -1.0, 2.0],
            ]
        )
    )
    return LinearMap(parent, parent, matrix)


def test_assembled_system_admits_exact_operator_rhs_contract() -> None:
    parent = _disk().cochain_space(0)
    operator = _cycle_operator(parent)
    rhs = parent.form(np.array([1.0, 2.0, 3.0, 4.0]), ORDINARY_FORM)

    system = AssembledSystem(operator, rhs)

    assert system.operator is operator
    assert system.rhs is rhs

    foreign = _disk().cochain_space(0)
    with pytest.raises(SystemError, match="equation space"):
        AssembledSystem(operator, foreign.form(np.ones(4), ORDINARY_FORM))


def test_dirichlet_elimination_uses_blocks_and_reconstructs_exact_values() -> None:
    parent = _disk().cochain_space(0)
    system = AssembledSystem(
        _cycle_operator(parent),
        parent.form(np.array([1.0, 2.0, 3.0, 4.0]), ORDINARY_FORM),
    )
    region = CochainSubspace(parent, np.array([0, 3], dtype=np.int64))
    values = region.form(np.array([10.0, 20.0]), ORDINARY_FORM)

    problem = eliminate_dirichlet(system, region, values)

    assert isinstance(problem, DirichletProblem)
    assert problem.parent is parent
    assert problem.boundary is region
    assert problem.interior.parent is parent
    np.testing.assert_array_equal(problem.interior.indices(), [1, 2])
    np.testing.assert_array_equal(
        problem.operator.matrix().toarray(),
        np.array([[2.0, -1.0], [-1.0, 2.0]]),
    )
    np.testing.assert_array_equal(problem.rhs.coefficients(), [12.0, 23.0])

    interior = problem.interior.form(np.array([5.0, 6.0]), ORDINARY_FORM)
    full = problem.reconstruct(interior)
    np.testing.assert_array_equal(full.coefficients(), [10.0, 5.0, 6.0, 20.0])
    assert full.space is parent
    assert full.semantics is interior.semantics


def test_dirichlet_region_must_lie_on_canonical_topological_boundary() -> None:
    domain = _disk()
    parent = domain.cochain_space(1)
    identity = LinearMap(parent, parent, csr_array(np.eye(parent.size)))
    system = AssembledSystem(identity, parent.form(np.ones(parent.size), ORDINARY_FORM))
    interior_edge = CochainSubspace(parent, np.array([1], dtype=np.int64))

    with pytest.raises(SystemError, match="topological boundary"):
        eliminate_dirichlet(
            system,
            interior_edge,
            interior_edge.form(np.zeros(1), ORDINARY_FORM),
        )


def test_dirichlet_elimination_requires_exact_parent_and_value_space() -> None:
    domain = _disk()
    parent = domain.cochain_space(0)
    system = AssembledSystem(
        _cycle_operator(parent), parent.form(np.ones(4), ORDINARY_FORM)
    )
    equivalent_parent = domain.cochain_space(0)
    foreign_region = CochainSubspace(equivalent_parent, np.array([0], dtype=np.int64))
    exact_region = CochainSubspace(parent, np.array([0], dtype=np.int64))
    wrong_value_space = CochainSubspace(
        equivalent_parent, np.array([1], dtype=np.int64)
    )

    with pytest.raises(SystemError, match="exact unknown space"):
        eliminate_dirichlet(
            system,
            foreign_region,
            foreign_region.form(np.zeros(1), ORDINARY_FORM),
        )
    with pytest.raises(SystemError, match="boundary value space"):
        eliminate_dirichlet(
            system,
            exact_region,
            wrong_value_space.form(np.zeros(1), ORDINARY_FORM),
        )


def test_empty_dirichlet_region_is_identity_reduction() -> None:
    parent = _disk().cochain_space(0)
    operator = _cycle_operator(parent)
    rhs = parent.form(np.arange(4, dtype=np.float64), ORDINARY_FORM)
    empty = CochainSubspace(parent, np.array([], dtype=np.int64))

    problem = eliminate_dirichlet(
        AssembledSystem(operator, rhs),
        empty,
        empty.form(np.array([], dtype=np.float64), ORDINARY_FORM),
    )

    np.testing.assert_array_equal(problem.interior.indices(), np.arange(4))
    np.testing.assert_array_equal(
        problem.operator.matrix().toarray(), operator.matrix().toarray()
    )
    np.testing.assert_array_equal(problem.rhs.coefficients(), rhs.coefficients())
    candidate = problem.interior.form(np.arange(4, dtype=np.float64), ORDINARY_FORM)
    np.testing.assert_array_equal(
        problem.reconstruct(candidate).coefficients(), candidate.coefficients()
    )


def test_all_unknowns_prescribed_has_valid_empty_reduced_problem() -> None:
    parent = _disk().cochain_space(0)
    region = CochainSubspace(parent, np.arange(parent.size, dtype=np.int64))
    values = region.form(np.array([4.0, 3.0, 2.0, 1.0]), ORDINARY_FORM)
    problem = eliminate_dirichlet(
        AssembledSystem(
            _cycle_operator(parent), parent.form(np.ones(4), ORDINARY_FORM)
        ),
        region,
        values,
    )

    assert problem.operator.matrix().shape == (0, 0)
    assert problem.rhs.coefficients().size == 0
    full = problem.reconstruct(
        problem.interior.form(np.array([], dtype=np.float64), ORDINARY_FORM)
    )
    np.testing.assert_array_equal(full.coefficients(), values.coefficients())


def test_reconstruction_rejects_foreign_interior_space() -> None:
    parent = _disk().cochain_space(0)
    region = CochainSubspace(parent, np.array([0], dtype=np.int64))
    problem = eliminate_dirichlet(
        AssembledSystem(
            _cycle_operator(parent), parent.form(np.ones(4), ORDINARY_FORM)
        ),
        region,
        region.form(np.zeros(1), ORDINARY_FORM),
    )
    foreign = CochainSubspace(parent, np.array([0, 2, 3], dtype=np.int64))

    with pytest.raises(SystemError, match="interior space"):
        problem.reconstruct(foreign.form(np.zeros(3), ORDINARY_FORM))


def test_dirichlet_reduction_recovers_representable_overflow_cancellation() -> None:
    parent = _disk().cochain_space(0)
    matrix = np.eye(parent.size)
    matrix[1, 0] = 1.0e308
    region = CochainSubspace(parent, np.array([0], dtype=np.int64))
    problem = eliminate_dirichlet(
        AssembledSystem(
            LinearMap(parent, parent, csr_array(matrix)),
            parent.form(np.array([0.0, 1.5e308, 0.0, 0.0]), ORDINARY_FORM),
        ),
        region,
        region.form(np.array([2.0]), ORDINARY_FORM),
    )

    assert problem.rhs.coefficients()[0] == pytest.approx(-5.0e307)


def test_dirichlet_reduction_recovers_finite_catastrophic_cancellation() -> None:
    parent = _disk().cochain_space(0)
    matrix = np.eye(parent.size)
    matrix[1, 0] = 1.0e200
    region = CochainSubspace(parent, np.array([0], dtype=np.int64))
    problem = eliminate_dirichlet(
        AssembledSystem(
            LinearMap(parent, parent, csr_array(matrix)),
            parent.form(np.array([0.0, 1.0, 0.0, 0.0]), ORDINARY_FORM),
        ),
        region,
        region.form(np.array([1.0e-200]), ORDINARY_FORM),
    )

    assert problem.rhs.coefficients()[0] == 4.816661538840688e-17


def test_dirichlet_reduction_recovers_cancellation_between_overflowing_products() -> (
    None
):
    parent = _disk().cochain_space(0)
    matrix = np.eye(parent.size)
    matrix[1, 0] = 1.0e308
    matrix[1, 3] = -1.0e308
    region = CochainSubspace(parent, np.array([0, 3], dtype=np.int64))
    problem = eliminate_dirichlet(
        AssembledSystem(
            LinearMap(parent, parent, csr_array(matrix)),
            parent.form(np.array([0.0, 1.0, 0.0, 0.0]), ORDINARY_FORM),
        ),
        region,
        region.form(np.array([1.0e308, 1.0e308]), ORDINARY_FORM),
    )

    assert problem.rhs.coefficients()[0] == 1.0


def test_dirichlet_reduction_rejects_exact_nonzero_underflow() -> None:
    parent = _disk().cochain_space(0)
    matrix = np.eye(parent.size)
    matrix[1, 0] = np.nextafter(0.0, 1.0)
    region = CochainSubspace(parent, np.array([0], dtype=np.int64))

    with pytest.raises(SystemError, match="unrepresentable"):
        eliminate_dirichlet(
            AssembledSystem(
                LinearMap(parent, parent, csr_array(matrix)),
                parent.form(np.zeros(parent.size), ORDINARY_FORM),
            ),
            region,
            region.form(np.array([0.5]), ORDINARY_FORM),
        )


def test_dirichlet_reduction_rejects_unrepresentable_rhs() -> None:
    parent = _disk().cochain_space(0)
    matrix = np.eye(parent.size)
    matrix[1, 0] = 1.0e308
    region = CochainSubspace(parent, np.array([0], dtype=np.int64))

    with pytest.raises(SystemError, match="unrepresentable"):
        eliminate_dirichlet(
            AssembledSystem(
                LinearMap(parent, parent, csr_array(matrix)),
                parent.form(np.zeros(parent.size), ORDINARY_FORM),
            ),
            region,
            region.form(np.array([2.0]), ORDINARY_FORM),
        )


def test_dirichlet_problem_has_no_public_construction_bypass() -> None:
    with pytest.raises(SystemError, match="eliminate_dirichlet"):
        DirichletProblem()


def test_dirichlet_elimination_and_reconstruction_preserve_semantic_class() -> None:
    class AlternateSemantics(FieldSemantics):
        pass

    parent = _disk().cochain_space(0)
    region = CochainSubspace(parent, np.array([0], dtype=np.int64))
    alternate = AlternateSemantics()
    system = AssembledSystem(
        _cycle_operator(parent),
        parent.form(np.ones(parent.size), ORDINARY_FORM),
    )

    with pytest.raises(SystemError, match="field semantics"):
        getattr(__import__("polygeo"), "eliminate_dirichlet")(
            system,
            region,
            region.form(np.zeros(1), alternate),
        )

    problem = eliminate_dirichlet(
        system,
        region,
        region.form(np.zeros(1), ORDINARY_FORM),
    )
    with pytest.raises(SystemError, match="field semantics"):
        getattr(problem, "reconstruct")(
            problem.interior.form(np.zeros(problem.interior.size), alternate)
        )
