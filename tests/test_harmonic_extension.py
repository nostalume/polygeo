from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from polygeo import (
    ORDINARY_FORM,
    AlgorithmError,
    CochainSubspace,
    Complex,
    FieldSemantics,
    Form,
    Geometry,
    LinearSolution,
    PositiveHodgeMetric,
    harmonic_extension,
    prepare_direct,
    topological_boundary,
)


class AlternateLinearSemantics(FieldSemantics):
    pass


def _interval(scale: float = 1.0) -> Geometry:
    domain = (
        Complex.from_maximal_simplices(np.array([[0, 1], [1, 2]], dtype=np.int64))
        .codimension_one_regular()
        .with_boundary()
        .connected()
    )
    return Geometry.from_positions(
        domain, scale * np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    )


def _simplex(dimension: int) -> Geometry:
    domain = (
        Complex.from_maximal_simplices(
            np.arange(dimension + 1, dtype=np.int64)[None, :]
        )
        .codimension_one_regular()
        .with_boundary()
        .connected()
    )
    return Geometry.from_positions(domain, np.eye(dimension + 1, dtype=np.float64))


def _annulus() -> Geometry:
    count = 5
    angles = 2.0 * math.pi * np.arange(count) / count
    positions = np.vstack(
        (
            np.column_stack((np.cos(angles), np.sin(angles))),
            3.3 * np.column_stack((np.cos(angles), np.sin(angles))),
        )
    )
    faces = np.array(
        [
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 7],
            [1, 7, 2],
            [2, 7, 8],
            [2, 8, 3],
            [3, 8, 4],
            [4, 8, 9],
            [4, 9, 5],
            [4, 5, 0],
        ],
        dtype=np.int64,
    )
    domain = (
        Complex.from_maximal_simplices(faces)
        .codimension_one_regular()
        .with_boundary()
        .connected()
    )
    return Geometry.from_positions(domain, positions)


def _boundary_values(
    geometry: Geometry, coefficients: np.ndarray, semantics=ORDINARY_FORM
) -> Form:
    parent = geometry.complex.cochain_space(0)
    indices = np.flatnonzero(topological_boundary(geometry.complex).mask(0)).astype(
        np.int64
    )
    return CochainSubspace(parent, indices).form(coefficients, semantics)


def test_harmonic_extension_restores_interval_boundary_and_harmonic_interior() -> None:
    geometry = _interval()
    metric = PositiveHodgeMetric(geometry)
    boundary = _boundary_values(geometry, np.array([0.0, 2.0]))

    solution = harmonic_extension(metric, boundary, prepare_direct)

    assert solution.form.space.complex is geometry.complex
    assert solution.equation_space.parent.complex is geometry.complex
    np.testing.assert_allclose(solution.form.coefficients(), [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        solution.form.coefficients()[boundary.space.indices()], boundary.coefficients()
    )
    operator = geometry.complex.boundary_matrix(1).toarray()
    stiffness = operator @ np.diag(metric.weights(1)) @ operator.T
    assert abs((stiffness @ solution.form.coefficients())[1]) <= 1.0e-14
    assert solution.relative_residual <= np.sqrt(np.finfo(np.float64).eps)


def test_harmonic_extension_preserves_constant_data() -> None:
    geometry = _interval()
    solution = harmonic_extension(
        PositiveHodgeMetric(geometry),
        _boundary_values(geometry, np.array([3.5, 3.5])),
        prepare_direct,
    )
    np.testing.assert_allclose(
        solution.form.coefficients(), 3.5, rtol=0.0, atol=2.0e-15
    )


@pytest.mark.parametrize("scale", [1.0e-100, 1.0e100])
def test_harmonic_extension_is_metric_scale_covariant(scale: float) -> None:
    baseline_geometry = _interval()
    scaled_geometry = _interval(scale)
    baseline = harmonic_extension(
        PositiveHodgeMetric(baseline_geometry),
        _boundary_values(baseline_geometry, np.array([0.0, 2.0])),
        prepare_direct,
    )
    scaled = harmonic_extension(
        PositiveHodgeMetric(scaled_geometry),
        _boundary_values(scaled_geometry, np.array([0.0, 2.0])),
        prepare_direct,
    )
    np.testing.assert_allclose(
        scaled.form.coefficients(),
        baseline.form.coefficients(),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_harmonic_extension_all_boundary_skips_backend_in_2d_and_3d() -> None:
    for dimension in (2, 3):
        geometry = _simplex(dimension)
        metric = PositiveHodgeMetric(geometry)
        values = np.arange(dimension + 1, dtype=np.float64)
        boundary = _boundary_values(geometry, values)
        calls = 0

        def forbidden_prepare(operator):
            nonlocal calls
            calls += 1
            raise AssertionError(operator)

        solution = harmonic_extension(metric, boundary, forbidden_prepare)

        assert calls == 0
        assert solution.equation_space.size == 0
        assert solution.residual_norm == 0.0
        assert solution.residual_scale == 0.0
        np.testing.assert_array_equal(solution.form.coefficients(), values)


def test_harmonic_extension_accepts_annulus_without_disk_evidence() -> None:
    geometry = _annulus()
    metric = PositiveHodgeMetric(geometry)
    boundary = _boundary_values(
        geometry,
        geometry.positions[
            np.flatnonzero(topological_boundary(geometry.complex).mask(0))
        ][:, 0],
    )
    solution = harmonic_extension(metric, boundary, prepare_direct)
    np.testing.assert_array_equal(
        solution.form.coefficients()[boundary.space.indices()], boundary.coefficients()
    )


def test_harmonic_extension_rejects_foreign_partial_and_reordered_boundary() -> None:
    geometry = _interval()
    metric = PositiveHodgeMetric(geometry)
    foreign = _interval()
    foreign_values = _boundary_values(foreign, np.array([0.0, 2.0]))
    with pytest.raises(AlgorithmError, match="different complex"):
        harmonic_extension(metric, cast(Any, foreign_values), prepare_direct)

    parent = geometry.complex.cochain_space(0)
    partial = CochainSubspace(parent, np.array([0], dtype=np.int64)).form(
        np.array([0.0]), ORDINARY_FORM
    )
    with pytest.raises(AlgorithmError, match="canonical boundary"):
        harmonic_extension(metric, cast(Any, partial), prepare_direct)

    reordered_space = object.__new__(CochainSubspace)
    reordered_space._parent = parent
    reordered_space._indices = np.array([2, 0], dtype=np.int64)
    reordered = Form(reordered_space, np.array([2.0, 0.0]), ORDINARY_FORM)
    with pytest.raises(AlgorithmError, match="canonical boundary"):
        harmonic_extension(metric, cast(Any, reordered), prepare_direct)


def test_harmonic_extension_preserves_generic_field_semantics() -> None:
    geometry = _interval()
    metric = PositiveHodgeMetric(geometry)
    semantics = AlternateLinearSemantics()
    values = _boundary_values(geometry, np.array([0.0, 2.0]), semantics)
    solution = harmonic_extension(metric, values, prepare_direct)
    assert solution.form.semantics is semantics
    np.testing.assert_allclose(solution.form.coefficients(), [0.0, 1.0, 2.0])


def test_harmonic_extension_closes_backend_and_foreign_evidence() -> None:
    geometry = _interval()
    metric = PositiveHodgeMetric(geometry)
    boundary = _boundary_values(geometry, np.array([0.0, 2.0]))

    def fail(operator):
        del operator
        raise ValueError("private injected text")

    with pytest.raises(
        AlgorithmError, match="harmonic extension solve failed"
    ) as caught:
        harmonic_extension(metric, boundary, fail)
    assert "private injected text" not in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)

    def malformed(operator):
        del operator
        return None

    with pytest.raises(AlgorithmError, match="solve failed"):
        harmonic_extension(metric, boundary, cast(Any, malformed))

    def private_failure(operator):
        del operator
        raise KeyError("PRIVATE_KEY")

    with pytest.raises(AlgorithmError, match="solve failed") as private:
        harmonic_extension(metric, boundary, cast(Any, private_failure))
    assert "PRIVATE_KEY" not in str(private.value)

    def forged_prepare(operator):
        def forged_solve(rhs):
            return LinearSolution(
                operator.source.form(np.zeros(operator.source.size), rhs.semantics),
                operator.target,
                0.0,
                0.0,
                0.0,
            )

        return forged_solve

    with pytest.raises(AlgorithmError, match="solve failed"):
        harmonic_extension(metric, boundary, forged_prepare)

    foreign = _interval()
    foreign_parent = foreign.complex.cochain_space(0)
    foreign_interior = CochainSubspace(foreign_parent, np.array([1], dtype=np.int64))

    def wrong_prepare(operator):
        solve = prepare_direct(operator)

        def wrong_solve(rhs):
            solution = solve(rhs)
            return LinearSolution(
                solution.form,
                foreign_interior,
                solution.residual_norm,
                solution.residual_scale,
                solution.relative_residual,
            )

        return wrong_solve

    with pytest.raises(AlgorithmError, match="foreign equation space"):
        harmonic_extension(metric, boundary, wrong_prepare)
