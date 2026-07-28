from __future__ import annotations

import numpy as np
import pytest

from polygeo import (
    Complex,
    FaceVectors,
    Geometry,
    PositiveHodgeMetric,
    SurfaceError,
    VertexVectors,
    face_unit_normals,
    mean_curvature_vectors,
    sphere_inscribed_vertex_normals,
    surface_area_gradient,
    tip_angle_vertex_normals,
    uniform_vertex_normals,
    volume_gradient,
)


def _tetrahedron(scale: float = 1.0):
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    domain = (
        Complex.from_maximal_simplices(faces)
        .triangle_manifold()
        .oriented()
        .without_boundary()
        .connected()
    )
    positions = scale * np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return Geometry.from_positions(domain, positions)


def _area(geometry: Geometry) -> float:
    return float(np.sum(geometry.primal_measures(2)))


def _signed_volume(geometry: Geometry) -> float:
    faces = geometry.complex.simplices(2)
    signs = geometry.complex.orientations(2)
    points = geometry.positions[faces]
    return float(
        np.sum(
            signs
            * np.einsum(
                "ij,ij->i",
                points[:, 0],
                np.cross(points[:, 1], points[:, 2]),
            )
        )
        / 6.0
    )


def _finite_difference(
    geometry: Geometry,
    functional,
    vertex: int,
    coordinate: int,
    step: float = 1e-6,
) -> float:
    positions = geometry.positions
    plus = positions.copy()
    minus = positions.copy()
    plus[vertex, coordinate] += step
    minus[vertex, coordinate] -= step
    return (
        functional(Geometry.from_positions(geometry.complex, plus))
        - functional(Geometry.from_positions(geometry.complex, minus))
    ) / (2.0 * step)


def test_surface_vector_outputs_bind_geometry_and_own_values() -> None:
    geometry = _tetrahedron()
    vertex_values = np.ones((4, 3), dtype=np.float64)
    face_values = np.ones((4, 3), dtype=np.float64)
    vertices = VertexVectors(geometry, vertex_values)
    faces = FaceVectors(geometry, face_values)

    assert vertices.geometry is geometry
    assert faces.geometry is geometry
    vertex_values[:] = 9.0
    face_values[:] = 9.0
    np.testing.assert_array_equal(vertices.vectors, np.ones((4, 3)))
    np.testing.assert_array_equal(faces.vectors, np.ones((4, 3)))

    normalized = vertices.normalized()
    np.testing.assert_allclose(np.linalg.norm(normalized.vectors, axis=1), 1.0)


def test_vector_outputs_reject_wrong_shape_and_zero_normalization() -> None:
    geometry = _tetrahedron()
    with pytest.raises(SurfaceError, match="one vector per vertex"):
        VertexVectors(geometry, np.ones((3, 3), dtype=np.float64))
    with pytest.raises(SurfaceError, match="zero vectors"):
        VertexVectors(geometry, np.zeros((4, 3), dtype=np.float64)).normalized()


def test_face_unit_normals_are_orthonormal_to_triangle_edges() -> None:
    geometry = _tetrahedron()
    normals = face_unit_normals(geometry)
    faces = geometry.complex.simplices(2)
    points = geometry.positions[faces]

    assert normals.geometry is geometry
    np.testing.assert_allclose(np.linalg.norm(normals.vectors, axis=1), 1.0)
    np.testing.assert_allclose(
        np.einsum("ij,ij->i", normals.vectors, points[:, 1] - points[:, 0]),
        0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.einsum("ij,ij->i", normals.vectors, points[:, 2] - points[:, 0]),
        0.0,
        atol=1e-14,
    )


def test_area_and_volume_gradients_match_finite_differences() -> None:
    geometry = _tetrahedron()
    area = surface_area_gradient(geometry).vectors
    volume = volume_gradient(geometry).vectors

    for vertex, coordinate in ((0, 0), (1, 2), (3, 1)):
        assert area[vertex, coordinate] == pytest.approx(
            _finite_difference(geometry, _area, vertex, coordinate),
            rel=2e-8,
            abs=2e-8,
        )
        assert volume[vertex, coordinate] == pytest.approx(
            _finite_difference(geometry, _signed_volume, vertex, coordinate),
            rel=2e-8,
            abs=2e-8,
        )


def test_weighted_vertex_normal_definitions_are_unit_and_coherently_oriented() -> None:
    geometry = _tetrahedron()
    expected = (
        np.sign(_signed_volume(geometry))
        * geometry.positions
        / np.linalg.norm(geometry.positions, axis=1)[:, None]
    )
    fields = (
        volume_gradient(geometry).normalized(),
        uniform_vertex_normals(geometry),
        tip_angle_vertex_normals(geometry),
        sphere_inscribed_vertex_normals(geometry),
    )
    for field in fields:
        np.testing.assert_allclose(np.linalg.norm(field.vectors, axis=1), 1.0)
        np.testing.assert_allclose(field.vectors, expected, atol=1e-14)


def test_mean_curvature_vectors_are_mass_normalized_area_gradient() -> None:
    geometry = _tetrahedron()
    metric = PositiveHodgeMetric(geometry)
    mean = mean_curvature_vectors(metric)
    area = surface_area_gradient(geometry)

    np.testing.assert_allclose(
        metric.weights(0)[:, None] * mean.vectors,
        area.vectors,
        rtol=2e-14,
        atol=2e-14,
    )


def test_normal_scale_laws() -> None:
    base = _tetrahedron()
    scaled = _tetrahedron(3.0)

    np.testing.assert_allclose(
        surface_area_gradient(scaled).vectors,
        3.0 * surface_area_gradient(base).vectors,
    )
    np.testing.assert_allclose(
        volume_gradient(scaled).vectors,
        9.0 * volume_gradient(base).vectors,
    )
    np.testing.assert_allclose(
        mean_curvature_vectors(PositiveHodgeMetric(scaled)).vectors,
        mean_curvature_vectors(PositiveHodgeMetric(base)).vectors / 3.0,
    )


def test_normal_computations_are_scale_safe_at_large_representable_scale() -> None:
    base = _tetrahedron()
    scaled = _tetrahedron(1e150)
    outward = base.positions / np.linalg.norm(base.positions, axis=1)[:, None]
    oriented = np.sign(_signed_volume(scaled)) * outward

    np.testing.assert_allclose(
        face_unit_normals(scaled).vectors, face_unit_normals(base).vectors
    )
    np.testing.assert_allclose(
        surface_area_gradient(scaled).normalized().vectors, outward
    )
    np.testing.assert_allclose(volume_gradient(scaled).normalized().vectors, oriented)
    np.testing.assert_allclose(
        sphere_inscribed_vertex_normals(scaled).vectors, oriented
    )


def test_surface_vector_normalization_is_scale_safe_for_huge_finite_vectors() -> None:
    geometry = _tetrahedron()
    directions = (
        geometry.positions / np.linalg.norm(geometry.positions, axis=1)[:, None]
    )
    normalized = VertexVectors(geometry, directions * 1e308).normalized()
    np.testing.assert_allclose(normalized.vectors, directions)
