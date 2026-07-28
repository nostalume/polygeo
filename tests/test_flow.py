from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from polygeo import (
    ORDINARY_FORM,
    AlgorithmError,
    Certified,
    Complex,
    FrozenFlowEvidence,
    Geometry,
    LinearSolution,
    PositiveHodgeMetric,
    SurfaceError,
    VertexMap,
    mean_curvature_flow_step,
    prepare_direct,
    vertex_map,
)


def _tetrahedron(scale: float = 1.0, shift: float = 0.0) -> Geometry:
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    domain = (
        Complex.from_maximal_simplices(faces)
        .triangle_manifold()
        .without_boundary()
        .connected()
    )
    positions = shift + scale * np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=np.float64,
    )
    return Geometry.from_positions(domain, positions)


def _bounded_triangle() -> Geometry:
    domain = (
        Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
        .triangle_manifold()
        .with_boundary()
        .connected()
    )
    return Geometry.from_positions(
        domain,
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]],
            dtype=np.float64,
        ),
    )


def _cycle() -> Geometry:
    domain = (
        Complex.from_maximal_simplices(
            np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
        )
        .codimension_one_regular()
        .without_boundary()
        .connected()
    )
    return Geometry.from_positions(
        domain,
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
    )


def test_vertex_map_is_factory_only_exact_identity_and_dimension_checked() -> None:
    source = _tetrahedron()
    target = Geometry.from_positions(source.complex, 0.5 * source.positions)
    mapped = vertex_map(source, target, 3)

    assert isinstance(mapped, VertexMap)
    assert mapped.source is source
    assert mapped.target is target
    assert mapped.source.complex is mapped.target.complex
    assert mapped.target_dimension == 3
    with pytest.raises(AlgorithmError, match="created by vertex_map"):
        VertexMap()
    with pytest.raises(AlgorithmError, match="exact complex"):
        vertex_map(source, _tetrahedron(), 3)
    with pytest.raises(AlgorithmError, match="target dimension"):
        vertex_map(source, target, 2)
    with pytest.raises(AttributeError):
        mapped._target = source


def test_flow_matches_frozen_diagonal_mass_equation_and_evidence() -> None:
    geometry = _tetrahedron()
    retained = geometry.positions
    metric = PositiveHodgeMetric(geometry)
    calls = 0
    solves = 0

    def prepare(operator):
        nonlocal calls
        calls += 1
        prepared = prepare_direct(operator)

        def solve(rhs):
            nonlocal solves
            solves += 1
            return prepared(rhs)

        return solve

    result = mean_curvature_flow_step(metric, 0.1, prepare)

    assert isinstance(result, Certified)
    assert result.output.source is geometry
    assert result.output.target.complex is geometry.complex
    assert result.output.target_dimension == geometry.ambient_dimension
    assert calls == 1
    assert solves == geometry.ambient_dimension
    np.testing.assert_array_equal(geometry.positions, retained)
    assert result.evidence.time_step == 0.1
    assert len(result.evidence.solves) == geometry.ambient_dimension
    assert result.evidence.energy_after < result.evidence.energy_before
    assert result.evidence.centroid.residual_norm <= (
        result.evidence.centroid.limit * result.evidence.centroid.scale
    )

    mass = np.diag(metric.weights(0))
    derivative = geometry.complex.boundary_matrix(1).toarray().T
    stiffness = derivative.T @ np.diag(metric.weights(1)) @ derivative
    np.testing.assert_allclose(
        (mass + 0.1 * stiffness) @ result.output.target.positions,
        mass @ geometry.positions,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    before = 0.5 * float(
        np.trace(geometry.positions.T @ stiffness @ geometry.positions)
    )
    target = result.output.target.positions
    after = 0.5 * float(np.trace(target.T @ stiffness @ target))
    assert result.evidence.energy_before == pytest.approx(before, rel=2.0e-15)
    assert result.evidence.energy_after == pytest.approx(after, rel=2.0e-15)
    for coordinate, evidence in enumerate(result.evidence.solves):
        product = (mass + 0.1 * stiffness) @ target[:, coordinate]
        rhs = mass @ geometry.positions[:, coordinate]
        residual = float(np.max(np.abs(product - rhs), initial=0.0))
        scale = max(
            float(np.max(np.abs(product), initial=0.0)),
            float(np.max(np.abs(rhs), initial=0.0)),
        )
        assert residual <= evidence.limit * scale
        assert evidence.residual_norm <= evidence.limit * evidence.scale
    assert set(result.output.__slots__) == {
        "_sealed",
        "_source",
        "_target",
        "_target_dimension",
    }
    assert set(result.evidence.__slots__) == {
        "time_step",
        "energy_before",
        "energy_after",
        "centroid",
        "solves",
    }


@pytest.mark.parametrize("scale", [1.0e-100, 1.0e100])
def test_flow_is_scale_covariant_at_extreme_represented_scales(scale: float) -> None:
    baseline = mean_curvature_flow_step(
        PositiveHodgeMetric(_tetrahedron()), 0.1, prepare_direct
    ).output.target.positions
    scaled = mean_curvature_flow_step(
        PositiveHodgeMetric(_tetrahedron(scale)), 0.1 * scale * scale, prepare_direct
    ).output.target.positions
    np.testing.assert_allclose(scaled / scale, baseline, rtol=8.0e-13, atol=8.0e-13)


def test_flow_is_translation_covariant_and_centroid_preserving() -> None:
    shift = 1.0e12
    baseline_geometry = _tetrahedron()
    translated_geometry = _tetrahedron(shift=shift)
    baseline = mean_curvature_flow_step(
        PositiveHodgeMetric(baseline_geometry), 0.1, prepare_direct
    )
    translated = mean_curvature_flow_step(
        PositiveHodgeMetric(translated_geometry), 0.1, prepare_direct
    )
    np.testing.assert_allclose(
        translated.output.target.positions - shift,
        baseline.output.target.positions,
        rtol=0.0,
        atol=3.0e-4,
    )
    weights = PositiveHodgeMetric(translated_geometry).weights(0)
    np.testing.assert_allclose(
        weights @ translated.output.target.positions / np.sum(weights),
        weights @ translated_geometry.positions / np.sum(weights),
        rtol=0.0,
        atol=2.0e-4,
    )


@pytest.mark.parametrize("time_step", [0.0, -1.0, math.inf, math.nan])
def test_flow_rejects_invalid_time_step(time_step: float) -> None:
    with pytest.raises(SurfaceError, match="time step"):
        mean_curvature_flow_step(
            PositiveHodgeMetric(_tetrahedron()), time_step, prepare_direct
        )


def test_flow_rejects_boundary_nontriangle_and_disconnected_domains() -> None:
    with pytest.raises(SurfaceError, match="closed connected triangle manifold"):
        mean_curvature_flow_step(
            PositiveHodgeMetric(_bounded_triangle()), 0.1, prepare_direct
        )
    with pytest.raises(SurfaceError, match="closed connected triangle manifold"):
        mean_curvature_flow_step(PositiveHodgeMetric(_cycle()), 0.1, prepare_direct)

    first = _tetrahedron()
    faces = np.vstack((first.complex.simplices(2), first.complex.simplices(2) + 4))
    domain = (
        Complex.from_maximal_simplices(faces).triangle_manifold().without_boundary()
    )
    positions = np.vstack(
        (first.positions, first.positions + np.array([5.0, 0.0, 0.0]))
    )
    disconnected = Geometry.from_positions(domain, positions)
    with pytest.raises(SurfaceError, match="closed connected triangle manifold"):
        mean_curvature_flow_step(
            cast(Any, PositiveHodgeMetric(disconnected)), 0.1, prepare_direct
        )


def test_flow_closes_solver_failures_and_rejects_foreign_evidence() -> None:
    geometry = _tetrahedron()
    metric = PositiveHodgeMetric(geometry)

    def fail(operator):
        del operator
        raise ValueError("private injected text")

    with pytest.raises(SurfaceError, match="flow solve failed") as caught:
        mean_curvature_flow_step(metric, 0.1, fail)
    assert "private injected text" not in str(caught.value)

    def malformed(operator):
        del operator
        return None

    with pytest.raises(SurfaceError, match="flow solve failed"):
        mean_curvature_flow_step(metric, 0.1, cast(Any, malformed))

    def key_failure(operator):
        del operator
        raise KeyError("PRIVATE_KEY")

    def surface_failure(operator):
        del operator
        raise SurfaceError("PRIVATE_CALLBACK_TEXT")

    for failure in (key_failure, surface_failure):
        with pytest.raises(SurfaceError, match="flow solve failed") as private:
            mean_curvature_flow_step(metric, 0.1, cast(Any, failure))
        assert "PRIVATE" not in str(private.value)

    foreign_space = _tetrahedron().complex.cochain_space(0)

    def wrong_prepare(operator):
        prepared = prepare_direct(operator)

        def wrong_solve(rhs):
            solved = prepared(rhs)
            return LinearSolution(
                solved.form,
                foreign_space,
                solved.residual_norm,
                solved.residual_scale,
                solved.relative_residual,
            )

        return wrong_solve

    with pytest.raises(SurfaceError, match="flow solve failed"):
        mean_curvature_flow_step(metric, 0.1, wrong_prepare)

    coordinate = 0

    def forged_prepare(operator):
        def forged_solve(rhs):
            nonlocal coordinate
            values = 0.9 * geometry.positions[:, coordinate]
            coordinate += 1
            return LinearSolution(
                operator.source.form(values, rhs.semantics),
                operator.target,
                0.0,
                0.0,
                0.0,
            )

        return forged_solve

    with pytest.raises(SurfaceError, match="flow solve failed"):
        mean_curvature_flow_step(metric, 0.1, forged_prepare)


def test_flow_readmits_output_and_rejects_degenerate_solver_positions() -> None:
    geometry = _tetrahedron()
    metric = PositiveHodgeMetric(geometry)

    def degenerate_prepare(operator):
        def solve(rhs):
            return LinearSolution(
                operator.source.form(np.zeros(operator.source.size), ORDINARY_FORM),
                operator.target,
                0.0,
                0.0,
                0.0,
            )

        return solve

    with pytest.raises(SurfaceError, match="flow solve failed"):
        mean_curvature_flow_step(metric, 0.1, cast(Any, degenerate_prepare))


def test_flow_requires_positive_metric_capability() -> None:
    with pytest.raises(SurfaceError, match="positive Hodge metric"):
        mean_curvature_flow_step(cast(Any, _tetrahedron()), 0.1, prepare_direct)


def test_frozen_flow_evidence_validates_residual_elements() -> None:
    with pytest.raises(SurfaceError, match="residual evidence"):
        FrozenFlowEvidence(0.1, 1.0, 0.5, cast(Any, object()), ())
