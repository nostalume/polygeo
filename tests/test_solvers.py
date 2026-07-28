from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.sparse import csr_array

from polygeo import (
    ORDINARY_FORM,
    AssembledSystem,
    CochainSubspace,
    Complex,
    FieldSemantics,
    LinearMap,
    LinearSolution,
    LeastSquaresSolution,
    NumericalError,
    eliminate_dirichlet,
    prepare_direct,
    prepare_least_squares,
)


def _disk():
    return Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    ).codimension_one_regular()


def _rectangular_operator():
    target = _disk().cochain_space(0)
    source = CochainSubspace(target, np.array([0, 1], dtype=np.int64))
    matrix = csr_array(
        np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
            dtype=np.float64,
        )
    )
    return source, target, LinearMap(source, target, matrix)


def test_prepared_least_squares_certifies_normal_equations_not_ax_equals_b() -> None:
    source, target, operator = _rectangular_operator()
    rhs = target.form(np.array([2.0, -1.0, 4.0, 5.0]), ORDINARY_FORM)

    solution = prepare_least_squares(operator)(rhs)
    expected, *_ = np.linalg.lstsq(operator.matrix().toarray(), rhs.coefficients())

    assert isinstance(solution, LeastSquaresSolution)
    assert solution.form.space is source
    assert solution.equation_space is target
    assert solution.form.semantics is rhs.semantics
    np.testing.assert_allclose(solution.form.coefficients(), expected)
    assert not np.allclose(
        operator.matrix() @ solution.form.coefficients(), rhs.coefficients()
    )
    assert solution.relative_normal_residual <= np.sqrt(np.finfo(np.float64).eps)
    assert solution.condition_indicator <= solution.condition_limit
    with pytest.raises(FrozenInstanceError):
        setattr(solution, "condition_indicator", 0.0)


@pytest.mark.parametrize("scale", [1.0e-300, 1.0e300])
def test_prepare_least_squares_normal_evidence_is_scale_invariant(scale: float) -> None:
    source, target, operator = _rectangular_operator()
    scaled = LinearMap(source, target, operator.matrix() * scale)
    rhs = target.form(scale * np.array([2.0, -1.0, 0.5, 3.0]), ORDINARY_FORM)

    solution = prepare_least_squares(scaled)(rhs)
    expected, *_ = np.linalg.lstsq(
        operator.matrix().toarray(), np.array([2.0, -1.0, 0.5, 3.0])
    )

    np.testing.assert_allclose(
        solution.form.coefficients(), expected, rtol=1.0e-14, atol=0.0
    )
    assert solution.relative_normal_residual <= np.sqrt(np.finfo(np.float64).eps)


def test_prepared_least_squares_reuses_qr_and_preserves_semantics(monkeypatch) -> None:
    import polygeo.solvers as solvers_module

    source, target, operator = _rectangular_operator()
    calls = 0
    original = solvers_module.qr

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solvers_module, "qr", tracked)
    prepared = prepare_least_squares(operator)

    class AlternateSemantics(FieldSemantics):
        pass

    first = prepared(target.form(np.arange(4.0), ORDINARY_FORM))
    alternate = AlternateSemantics()
    second = prepared(target.form(np.arange(4.0), alternate))

    assert calls == 1
    assert first.form.semantics is ORDINARY_FORM
    assert second.form.semantics is alternate
    assert first.form.space is source


def test_prepare_least_squares_rejects_shape_rank_condition_and_foreign_rhs() -> None:
    source, target, operator = _rectangular_operator()
    with pytest.raises(NumericalError, match="rows"):
        prepare_least_squares(
            LinearMap(target, source, csr_array(np.ones((source.size, target.size))))
        )
    with pytest.raises(NumericalError, match="rank"):
        prepare_least_squares(
            LinearMap(source, target, csr_array(np.ones((target.size, source.size))))
        )
    ill = operator.matrix().toarray()
    ill[:, 1] = ill[:, 0] + 1.0e-12 * ill[:, 1]
    with pytest.raises(NumericalError, match="condition"):
        prepare_least_squares(LinearMap(source, target, csr_array(ill)))

    foreign = _disk().cochain_space(0)
    prepared = prepare_least_squares(operator)
    with pytest.raises(NumericalError, match="equation space"):
        prepared(foreign.form(np.ones(foreign.size), ORDINARY_FORM))


def test_prepare_least_squares_empty_source_calls_no_backend(monkeypatch) -> None:
    target = _disk().cochain_space(0)
    source = CochainSubspace(target, np.array([], dtype=np.int64))
    operator = LinearMap(source, target, csr_array((target.size, 0)))

    def forbidden(*args, **kwargs):
        raise AssertionError((args, kwargs))

    monkeypatch.setattr("polygeo.solvers.qr", forbidden)
    solution = prepare_least_squares(operator)(
        target.form(np.ones(target.size), ORDINARY_FORM)
    )

    assert solution.form.coefficients().size == 0
    assert solution.normal_residual_norm == 0.0
    assert solution.normal_residual_scale == 0.0
    assert solution.condition_indicator == 0.0


def test_prepare_least_squares_closes_backend_errors(monkeypatch) -> None:
    _, _, operator = _rectangular_operator()

    def fail(*args, **kwargs):
        raise ValueError("private backend text")

    monkeypatch.setattr("polygeo.solvers.qr", fail)
    with pytest.raises(NumericalError, match="least-squares preparation") as caught:
        prepare_least_squares(operator)
    assert "private backend text" not in str(caught.value)


def test_prepare_direct_solves_and_certifies_exact_endpoint_spaces() -> None:
    domain = _disk()
    source = domain.cochain_space(0)
    target = domain.cochain_space(0)
    matrix = csr_array(
        np.array(
            [
                [4.0, 1.0, 0.0, 0.0],
                [1.0, 3.0, 1.0, 0.0],
                [0.0, 1.0, 3.0, 1.0],
                [0.0, 0.0, 1.0, 2.0],
            ]
        )
    )
    operator = LinearMap(source, target, matrix)
    rhs = target.form(np.array([1.0, 2.0, 3.0, 4.0]), ORDINARY_FORM)

    solution = prepare_direct(operator)(rhs)

    assert isinstance(solution, LinearSolution)
    assert solution.form.space is source
    assert solution.equation_space is target
    assert solution.form.semantics is rhs.semantics
    np.testing.assert_allclose(
        matrix @ solution.form.coefficients(), rhs.coefficients()
    )
    assert solution.residual_norm >= 0.0
    assert solution.residual_scale > 0.0
    assert solution.relative_residual <= np.sqrt(np.finfo(np.float64).eps)
    with pytest.raises(FrozenInstanceError):
        setattr(solution, "residual_norm", 1.0)
    rebuilt = LinearSolution(
        solution.form,
        target,
        solution.residual_norm,
        solution.residual_scale,
        solution.relative_residual,
    )
    assert rebuilt == solution
    with pytest.raises(NumericalError, match="residual"):
        LinearSolution(solution.form, target, -1.0, 1.0, -1.0)
    with pytest.raises(NumericalError, match="residual"):
        LinearSolution(solution.form, target, 1.0, 0.0, 0.0)


def test_prepared_direct_reuses_factorization_for_multiple_semantics() -> None:
    class AlternateSemantics(FieldSemantics):
        pass

    parent = _disk().cochain_space(0)
    operator = LinearMap(parent, parent, csr_array(np.diag([1.0, 2.0, 3.0, 4.0])))
    prepared = prepare_direct(operator)
    first_rhs = parent.form(np.array([1.0, 2.0, 3.0, 4.0]), ORDINARY_FORM)
    alternate = AlternateSemantics()
    second_rhs = parent.form(np.array([2.0, 4.0, 6.0, 8.0]), alternate)

    first = prepared(first_rhs)
    second = prepared(second_rhs)

    np.testing.assert_array_equal(first.form.coefficients(), np.ones(4))
    np.testing.assert_array_equal(second.form.coefficients(), np.full(4, 2.0))
    assert first.form.semantics is first_rhs.semantics
    assert second.form.semantics is alternate


def test_prepared_direct_rejects_foreign_rhs_space() -> None:
    parent = _disk().cochain_space(0)
    foreign = _disk().cochain_space(0)
    prepared = prepare_direct(LinearMap(parent, parent, csr_array(np.eye(4))))

    with pytest.raises(NumericalError, match="equation space"):
        getattr(prepared, "__call__")(
            foreign.form(np.ones(foreign.size), ORDINARY_FORM)
        )


def test_prepare_direct_rejects_nonsquare_and_singular_operators() -> None:
    domain = _disk()
    vertices = domain.cochain_space(0)
    edges = domain.cochain_space(1)
    rectangular = LinearMap(
        vertices,
        edges,
        csr_array((edges.size, vertices.size)),
    )
    with pytest.raises(NumericalError, match="square"):
        prepare_direct(rectangular)

    singular = LinearMap(vertices, vertices, csr_array(np.ones((4, 4))))
    with pytest.raises(NumericalError, match="factor"):
        prepare_direct(singular)


def test_prepare_direct_supports_empty_system() -> None:
    parent = _disk().cochain_space(0)
    empty = CochainSubspace(parent, np.array([], dtype=np.int64))
    operator = LinearMap(empty, empty, csr_array((0, 0)))
    rhs = empty.form(np.array([], dtype=np.float64), ORDINARY_FORM)

    solution = prepare_direct(operator)(rhs)

    assert solution.form.coefficients().size == 0
    assert solution.residual_norm == 0.0
    assert solution.residual_scale == 0.0
    assert solution.relative_residual == 0.0


def test_direct_residual_evidence_is_scale_stable() -> None:
    parent = _disk().cochain_space(0)
    relative = []
    for magnitude in (1.0e-300, 1.0, 1.0e300):
        operator = LinearMap(
            parent,
            parent,
            csr_array(np.eye(parent.size) * magnitude),
        )
        rhs = parent.form(
            np.array([1.0, -2.0, 3.0, -4.0]) * magnitude,
            ORDINARY_FORM,
        )
        solution = prepare_direct(operator)(rhs)
        relative.append(solution.relative_residual)
        assert solution.residual_scale == 4.0 * magnitude

    assert all(value <= np.sqrt(np.finfo(np.float64).eps) for value in relative)


def test_direct_residual_rejects_cancellation_damaged_backend_solution() -> None:
    domain = Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    ).codimension_one_regular()
    parent = domain.cochain_space(0)
    operator = LinearMap(
        parent,
        parent,
        csr_array(
            np.array(
                [
                    [1.0e16, 1.0, -1.0e16],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        ),
    )
    rhs = parent.form(np.array([0.0, 1.0, 1.0]), ORDINARY_FORM)

    with pytest.raises(NumericalError, match="residual"):
        prepare_direct(operator)(rhs)


def test_direct_residual_rejects_exact_nonzero_underflow(monkeypatch) -> None:
    class UnderflowingFactor:
        def solve(self, rhs):
            return np.array([0.5, 0.0, 0.0, 0.0])

    monkeypatch.setattr("polygeo.solvers.splu", lambda matrix: UnderflowingFactor())
    parent = _disk().cochain_space(0)
    diagonal = np.eye(parent.size)
    diagonal[0, 0] = np.nextafter(0.0, 1.0)
    operator = LinearMap(parent, parent, csr_array(diagonal))
    rhs = parent.form(np.zeros(parent.size), ORDINARY_FORM)

    with pytest.raises(NumericalError, match="residual"):
        prepare_direct(operator)(rhs)


def test_prepare_direct_factors_once_and_hides_backend_errors(monkeypatch) -> None:
    calls = 0

    class IdentityFactor:
        def solve(self, rhs):
            return rhs.copy()

    def factor(matrix):
        nonlocal calls
        calls += 1
        return IdentityFactor()

    monkeypatch.setattr("polygeo.solvers.splu", factor)
    parent = _disk().cochain_space(0)
    operator = LinearMap(parent, parent, csr_array(np.eye(parent.size)))
    rhs = parent.form(np.ones(parent.size), ORDINARY_FORM)
    prepared = prepare_direct(operator)

    prepared(rhs)
    prepared(rhs)
    assert calls == 1

    def fail(matrix):
        raise ValueError("private backend text")

    monkeypatch.setattr("polygeo.solvers.splu", fail)
    with pytest.raises(NumericalError, match="factorization") as failure:
        prepare_direct(operator)
    assert "private backend text" not in str(failure.value)


def test_prepared_direct_hides_solve_backend_errors(monkeypatch) -> None:
    class FailingFactor:
        def solve(self, rhs):
            raise RuntimeError("private solve text")

    monkeypatch.setattr("polygeo.solvers.splu", lambda matrix: FailingFactor())
    parent = _disk().cochain_space(0)
    operator = LinearMap(parent, parent, csr_array(np.eye(parent.size)))
    rhs = parent.form(np.ones(parent.size), ORDINARY_FORM)
    prepared = prepare_direct(operator)

    with pytest.raises(NumericalError, match="direct solve failed") as failure:
        prepared(rhs)
    assert "private solve text" not in str(failure.value)


def test_assembled_system_solve_uses_injected_preparer() -> None:
    parent = _disk().cochain_space(0)
    system = AssembledSystem(
        LinearMap(parent, parent, csr_array(np.eye(parent.size))),
        parent.form(np.arange(parent.size, dtype=np.float64), ORDINARY_FORM),
    )
    seen = []

    def prepare(operator):
        seen.append(operator)
        return prepare_direct(operator)

    solution = system.solve(prepare)

    assert seen == [system.operator]
    np.testing.assert_array_equal(
        solution.form.coefficients(), system.rhs.coefficients()
    )


def test_dirichlet_problem_solve_reconstructs_and_keeps_reduced_evidence() -> None:
    parent = _disk().cochain_space(0)
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
    region = CochainSubspace(parent, np.array([0, 3], dtype=np.int64))
    problem = eliminate_dirichlet(
        AssembledSystem(
            LinearMap(parent, parent, matrix),
            parent.form(np.array([0.0, 2.0, 3.0, 0.0]), ORDINARY_FORM),
        ),
        region,
        region.form(np.array([10.0, 20.0]), ORDINARY_FORM),
    )

    reduced = prepare_direct(problem.operator)(problem.rhs)
    solution = problem.solve(prepare_direct)

    np.testing.assert_array_equal(
        solution.form.coefficients(),
        problem.reconstruct(reduced.form).coefficients(),
    )
    assert solution.residual_norm == reduced.residual_norm
    assert solution.residual_scale == reduced.residual_scale
    assert solution.relative_residual == reduced.relative_residual
