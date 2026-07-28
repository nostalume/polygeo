from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from polygeo import (
    PositiveHodgeMetric,
    SurfaceError,
    admit_integrable_connection,
    connection_holonomy,
    gaussian_curvature_measure,
    integral_dual_cycles,
    integrate_direction_field,
    triangle_frames,
)

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
_connections = import_module("support.connections")
_meshes = import_module("support.meshes")
annulus = _meshes.annulus
icosphere = _meshes.icosphere
torus = _meshes.torus
torus_connection = _connections.torus_connection


def test_icosphere_is_deterministic_positive_and_gauss_bonnet() -> None:
    first, geometry = icosphere(2)
    second, repeated = icosphere(2)
    assert first.simplex_count(0) == 162
    assert first.simplex_count(2) == 320
    np.testing.assert_array_equal(first.simplices(2), second.simplices(2))
    np.testing.assert_array_equal(geometry.positions, repeated.positions)
    PositiveHodgeMetric(geometry)
    np.testing.assert_allclose(
        np.sum(gaussian_curvature_measure(geometry).coefficients()),
        4.0 * np.pi,
        rtol=0.0,
        atol=2e-13,
    )


def test_torus_has_two_curvature_signs_and_zero_total() -> None:
    domain, geometry, minor_angles = torus(12, 16)
    curvature = gaussian_curvature_measure(geometry).coefficients()
    assert domain.simplex_count(0) == 192
    assert minor_angles.shape == (192,)
    assert np.min(curvature) < 0.0 < np.max(curvature)
    np.testing.assert_allclose(np.sum(curvature), 0.0, atol=2e-13)


@pytest.mark.parametrize("major, minor", [(4, 6), (6, 8), (8, 10)])
@pytest.mark.parametrize("case", ["levi-civita", "cancelled", "quarter-turn"])
def test_connection_notebook_presets_have_the_claimed_holonomy(
    major: int, minor: int, case: str
) -> None:
    _, geometry, _ = torus(major, minor)
    connection = torus_connection(geometry, case)
    cycles = integral_dual_cycles(geometry)
    evidence = connection_holonomy(connection, cycles)

    if case == "levi-civita":
        assert evidence.local_error > evidence.limit
        with pytest.raises(SurfaceError, match="not integrable"):
            admit_integrable_connection(connection, cycles)
    elif case == "cancelled":
        assert evidence.local_error <= evidence.limit
        assert evidence.generator_error <= evidence.limit
        assert admit_integrable_connection(connection, cycles).connection is connection
    else:
        faces = geometry.complex.simplices(2)
        centers = geometry.positions[faces].mean(axis=1)
        major_angles = np.arctan2(centers[:, 1], centers[:, 0])
        dual_edges = connection.dual_edges()
        raw = major_angles[dual_edges[:, 1]] - major_angles[dual_edges[:, 0]]
        winding = np.arctan2(np.sin(raw), np.cos(raw))
        np.testing.assert_allclose(
            connection.transport_products(),
            np.exp(0.25j * winding),
            rtol=0.0,
            atol=3e-15,
        )
        assert evidence.local_error <= evidence.limit
        assert evidence.generator_error > evidence.limit
        np.testing.assert_allclose(
            evidence.generator_error, np.pi / 2.0, rtol=0.0, atol=3e-15
        )
        with pytest.raises(SurfaceError, match="not integrable"):
            admit_integrable_connection(connection, cycles)


@pytest.mark.parametrize("major, minor", [(4, 6), (6, 8), (8, 10)])
@pytest.mark.parametrize("anchor_phase", [-3.0, 0.0, 3.0])
def test_connection_notebook_admitted_field_controls(
    major: int, minor: int, anchor_phase: float
) -> None:
    _, geometry, _ = torus(major, minor)
    connection = torus_connection(geometry, "cancelled")
    capability = admit_integrable_connection(connection, integral_dual_cycles(geometry))
    result = integrate_direction_field(capability, anchor_phase=anchor_phase)
    field = result.output
    frames = triangle_frames(geometry)
    vectors = field.vectors()

    assert field.geometry is geometry
    assert field.connection is connection
    assert field.anchor_face == 0
    assert field.anchor_phase == anchor_phase
    assert result.evidence.crossing_error <= result.evidence.limit
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=3e-15)
    np.testing.assert_allclose(
        np.einsum("ij,ij->i", vectors, frames.normals()), 0.0, atol=3e-15
    )


def test_annulus_is_positive_hodge_admitted_and_repeatable() -> None:
    first, geometry = annulus(4, 16)
    second, repeated = annulus(4, 16)
    assert first.simplex_count(0) == 64
    np.testing.assert_array_equal(first.simplices(2), second.simplices(2))
    np.testing.assert_array_equal(geometry.positions, repeated.positions)
    metric = PositiveHodgeMetric(geometry)
    assert all(np.min(metric.weights(k)) > 0.0 for k in range(3))


@pytest.mark.parametrize(
    ("rings", "sections"),
    [(rings, 4 * rings) for rings in range(3, 7)]
    + [(3, sections) for sections in range(12, 17)]
    + [(4, sections) for sections in range(14, 17)],
)
def test_annulus_notebook_control_presets_are_positive_hodge_admitted(
    rings: int, sections: int
) -> None:
    _, geometry = annulus(rings, sections)
    metric = PositiveHodgeMetric(geometry)
    assert all(np.min(metric.weights(k)) > 0.0 for k in range(3))
