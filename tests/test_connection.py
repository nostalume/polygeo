from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import null_space

from polygeo import (
    Complex,
    Geometry,
    SurfaceError,
    admit_integrable_connection,
    connection_holonomy,
    integral_dual_cycles,
    integrate_direction_field,
    levi_civita_connection,
    surface_connection,
    triangle_frames,
)


def _tetrahedron() -> Geometry:
    domain = (
        Complex.from_maximal_simplices(
            np.array(
                [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
                dtype=np.int64,
            )
        )
        .triangle_manifold()
        .oriented()
        .without_boundary()
        .connected()
    )
    return Geometry.from_positions(
        domain,
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )


def _torus(major_sections: int = 4, minor_sections: int = 5) -> Geometry:
    faces: list[tuple[int, int, int]] = []
    for major in range(major_sections):
        for minor in range(minor_sections):
            lower = major * minor_sections + minor
            major_next = ((major + 1) % major_sections) * minor_sections + minor
            diagonal = ((major + 1) % major_sections) * minor_sections + (
                minor + 1
            ) % minor_sections
            minor_next = major * minor_sections + (minor + 1) % minor_sections
            faces.extend(((lower, major_next, diagonal), (lower, diagonal, minor_next)))
    domain = (
        Complex.from_maximal_simplices(np.array(faces, dtype=np.int64))
        .triangle_manifold()
        .oriented()
        .without_boundary()
        .connected()
    )
    major_angles = np.repeat(
        2.0 * np.pi * np.arange(major_sections) / major_sections, minor_sections
    )
    minor_angles = np.tile(
        2.0 * np.pi * np.arange(minor_sections) / minor_sections, major_sections
    )
    radial = 2.0 + 0.6 * np.cos(minor_angles)
    positions = np.column_stack(
        (
            radial * np.cos(major_angles),
            radial * np.sin(major_angles),
            0.6 * np.sin(minor_angles),
        )
    ).astype(np.float64)
    return Geometry.from_positions(domain, positions)


def _dual_incidence(geometry: Geometry) -> NDArray[np.int64]:
    boundary = geometry.complex.boundary_matrix(2).toarray()
    incidence = np.zeros(
        (geometry.complex.simplex_count(2), geometry.complex.simplex_count(1)),
        dtype=np.int64,
    )
    for edge in range(incidence.shape[1]):
        source, target = np.flatnonzero(boundary[edge])
        incidence[source, edge] = -1
        incidence[target, edge] = 1
    return incidence


def _local_dual_cycles(geometry: Geometry) -> NDArray[np.int64]:
    edges = geometry.complex.simplices(1)
    boundary = geometry.complex.boundary_matrix(2).toarray()
    face_count = geometry.complex.simplex_count(2)
    cycles = np.zeros((len(edges), geometry.complex.vertex_count), dtype=np.int64)
    for vertex in range(geometry.complex.vertex_count):
        incident_edges = np.flatnonzero(np.any(edges == vertex, axis=1))
        adjacency: dict[int, list[tuple[int, int]]] = {}
        for edge in incident_edges:
            left, right = map(int, np.flatnonzero(boundary[edge]))
            adjacency.setdefault(left, []).append((right, int(edge)))
            adjacency.setdefault(right, []).append((left, int(edge)))
        start = min(adjacency)
        previous = -1
        current = start
        next_face, edge = min(adjacency[current])
        while True:
            source, target = sorted((current, next_face))
            cycles[edge, vertex] = 1 if (current, next_face) == (source, target) else -1
            previous, current = current, next_face
            if current == start:
                break
            candidates = [item for item in adjacency[current] if item[0] != previous]
            assert len(candidates) == 1
            next_face, edge = candidates[0]
        assert np.count_nonzero(cycles[:, vertex]) == len(incident_edges)
    assert np.array_equal(
        _dual_incidence(geometry) @ cycles,
        np.zeros((face_count, geometry.complex.vertex_count), dtype=np.int64),
    )
    return cycles


def _trivial_connection(geometry: Geometry):
    levi_civita = levi_civita_connection(geometry)
    return surface_connection(
        geometry,
        -np.angle(levi_civita.transport_products()).astype(np.float64),
    )


def _represented_cycle_products(
    transports: NDArray[np.complex128],
    coefficients: NDArray[np.int64],
    *,
    reverse: bool = False,
) -> tuple[complex, ...]:
    products: list[complex] = []
    for column in range(coefficients.shape[1]):
        edges = np.flatnonzero(coefficients[:, column])
        if reverse:
            edges = edges[::-1]
        product = 1.0 + 0.0j
        for edge in edges:
            product *= complex(transports[edge]) ** int(coefficients[edge, column])
            product /= abs(product)
        products.append(product)
    return tuple(products)


def test_triangle_frames_are_owned_orthonormal_and_right_handed() -> None:
    geometry = _tetrahedron()
    frames = triangle_frames(geometry)
    first = frames.first_axes()
    second = frames.second_axes()
    normals = frames.normals()

    assert frames.geometry is geometry
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(second, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    np.testing.assert_allclose(np.einsum("ij,ij->i", first, second), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.cross(first, second), normals, atol=1e-15)
    retained_first = first.copy()
    first[:] = 0.0
    np.testing.assert_array_equal(frames.first_axes(), retained_first)


def test_levi_civita_transport_reverses_by_group_inverse() -> None:
    connection = levi_civita_connection(_tetrahedron())
    dual_edges = connection.dual_edges()
    products = connection.transport_products()
    assert products.dtype == np.dtype(np.complex128)
    np.testing.assert_allclose(np.abs(products), 1.0, atol=2e-15)
    for (source, target), product in zip(dual_edges, products, strict=True):
        assert connection.transport(int(source), int(target)) == product
        assert connection.transport(int(target), int(source)) == np.conjugate(product)


def test_levi_civita_hinge_sign_maps_source_normal_to_target_normal() -> None:
    geometry = _tetrahedron()
    connection = levi_civita_connection(geometry)
    frames = triangle_frames(geometry)
    normals = frames.normals()
    first = frames.first_axes()
    second = frames.second_axes()
    products = connection.transport_products()
    edges = geometry.complex.simplices(1)
    positions = geometry.positions
    for edge, (source, target) in enumerate(connection.dual_edges()):
        axis = positions[edges[edge, 1]] - positions[edges[edge, 0]]
        axis /= np.linalg.norm(axis)
        sine = axis @ np.cross(normals[source], normals[target])
        cosine = normals[source] @ normals[target]
        angle = math.atan2(float(sine), float(cosine))
        rotated = (
            normals[source] * math.cos(angle)
            + np.cross(axis, normals[source]) * math.sin(angle)
            + axis * (axis @ normals[source]) * (1.0 - math.cos(angle))
        )
        np.testing.assert_allclose(rotated, normals[target], atol=2e-15)
        rotated_first = (
            first[source] * math.cos(angle)
            + np.cross(axis, first[source]) * math.sin(angle)
            + axis * (axis @ first[source]) * (1.0 - math.cos(angle))
        )
        expected = complex(
            rotated_first @ first[target], rotated_first @ second[target]
        )
        expected /= abs(expected)
        np.testing.assert_allclose(products[edge], expected, atol=2e-15)


def test_connection_retains_lifted_deviation_and_composes_in_so2() -> None:
    geometry = _tetrahedron()
    levi_civita = levi_civita_connection(geometry)
    deviations = np.linspace(-0.4, 0.5, len(levi_civita.dual_edges()), dtype=np.float64)
    connection = surface_connection(geometry, deviations)
    expected = levi_civita.transport_products() * np.exp(1j * deviations)
    np.testing.assert_allclose(connection.transport_products(), expected, atol=2e-15)
    np.testing.assert_array_equal(connection.deviation_angles(), deviations)
    deviations[:] = 0.0
    assert not np.all(connection.deviation_angles() == 0.0)


@pytest.mark.parametrize(("major_sections", "minor_sections"), [(4, 5), (5, 6)])
def test_integral_dual_cycles_are_exact_primitive_tree_cotree_generators(
    major_sections: int, minor_sections: int
) -> None:
    geometry = _torus(major_sections, minor_sections)
    first = integral_dual_cycles(geometry)
    second = integral_dual_cycles(geometry)
    coefficients = first.cycle_coefficients().toarray()

    assert first.geometry is geometry
    assert first.dimension == 2
    assert coefficients.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(coefficients, second.cycle_coefficients().toarray())
    np.testing.assert_array_equal(
        _dual_incidence(geometry) @ coefficients,
        np.zeros((geometry.complex.simplex_count(2), 2), dtype=np.int64),
    )
    for column in range(first.dimension):
        assert math.gcd(*np.abs(coefficients[:, column])) == 1
    generator_edges = first.generator_edge_indices()
    np.testing.assert_array_equal(
        coefficients[generator_edges], np.eye(first.dimension, dtype=np.int64)
    )


def test_sphere_has_no_integral_dual_generators() -> None:
    basis = integral_dual_cycles(_tetrahedron())
    assert basis.dimension == 0
    assert basis.cycle_coefficients().shape == (6, 0)


def test_local_holonomy_rejects_levi_civita_on_curved_tetrahedron() -> None:
    geometry = _tetrahedron()
    connection = levi_civita_connection(geometry)
    cycles = integral_dual_cycles(geometry)
    evidence = connection_holonomy(connection, cycles)
    assert evidence.local_error > evidence.limit
    assert evidence.generator_error == 0.0
    with pytest.raises(SurfaceError, match="not integrable"):
        admit_integrable_connection(connection, cycles)


def test_integrable_connection_authorizes_tangent_direction_field() -> None:
    geometry = _tetrahedron()
    connection = _trivial_connection(geometry)
    cycles = integral_dual_cycles(geometry)
    capability = admit_integrable_connection(connection, cycles)
    result = integrate_direction_field(capability, anchor_phase=0.3)
    field = result.output
    evidence = result.evidence

    assert capability.connection is connection
    assert field.geometry is geometry
    assert field.connection is connection
    assert field.anchor_face == 0
    assert field.anchor_phase == 0.3
    phases = field.phases()
    vectors = field.vectors()
    frames = triangle_frames(geometry)
    reconstructed = (
        np.cos(phases)[:, None] * frames.first_axes()
        + np.sin(phases)[:, None] * frames.second_axes()
    )
    np.testing.assert_allclose(vectors, reconstructed, atol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-15)
    np.testing.assert_allclose(
        np.einsum("ij,ij->i", vectors, frames.normals()), 0.0, atol=2e-15
    )
    assert evidence.crossing_error <= evidence.limit


def test_global_phase_rotates_every_face_vector_in_its_tangent_plane() -> None:
    geometry = _tetrahedron()
    capability = admit_integrable_connection(
        _trivial_connection(geometry), integral_dual_cycles(geometry)
    )
    zero = integrate_direction_field(capability, anchor_phase=0.0).output
    quarter = integrate_direction_field(capability, anchor_phase=np.pi / 2.0).output
    frames = triangle_frames(geometry)
    expected = (
        -np.sin(zero.phases())[:, None] * frames.first_axes()
        + np.cos(zero.phases())[:, None] * frames.second_axes()
    )
    np.testing.assert_allclose(quarter.vectors(), expected, atol=3e-15)


def test_rigid_motion_covariance_of_frames_transport_and_field() -> None:
    geometry = _tetrahedron()
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    shift = np.array([3.0, -2.0, 5.0], dtype=np.float64)
    transformed = Geometry.from_positions(
        geometry.complex, geometry.positions @ rotation.T + shift
    )
    base_frames = triangle_frames(geometry)
    moved_frames = triangle_frames(transformed)
    np.testing.assert_allclose(
        moved_frames.first_axes(), base_frames.first_axes() @ rotation.T, atol=2e-15
    )
    np.testing.assert_allclose(
        moved_frames.second_axes(), base_frames.second_axes() @ rotation.T, atol=2e-15
    )
    np.testing.assert_allclose(
        moved_frames.normals(), base_frames.normals() @ rotation.T, atol=2e-15
    )
    np.testing.assert_allclose(
        levi_civita_connection(transformed).transport_products(),
        levi_civita_connection(geometry).transport_products(),
        atol=2e-15,
    )

    base_field = integrate_direction_field(
        admit_integrable_connection(
            _trivial_connection(geometry), integral_dual_cycles(geometry)
        ),
        anchor_phase=0.4,
    ).output
    moved_field = integrate_direction_field(
        admit_integrable_connection(
            _trivial_connection(transformed), integral_dual_cycles(transformed)
        ),
        anchor_phase=0.4,
    ).output
    np.testing.assert_allclose(moved_field.phases(), base_field.phases(), atol=2e-15)
    np.testing.assert_allclose(
        moved_field.vectors(), base_field.vectors() @ rotation.T, atol=3e-15
    )


def test_generator_holonomy_rejects_locally_flat_torus_connection() -> None:
    geometry = _torus()
    cycles = integral_dual_cycles(geometry)
    generators = cycles.cycle_coefficients().toarray().astype(np.float64)
    local = _local_dual_cycles(geometry).astype(np.float64)
    kernel = null_space(local.T)
    projected = kernel @ (kernel.T @ generators[:, 0])
    assert abs(generators[:, 0] @ projected) > 1e-8
    target_angles = 0.2 * projected / np.max(np.abs(projected))
    levi_civita = levi_civita_connection(geometry)
    deviations = np.angle(
        np.exp(1j * target_angles) * np.conjugate(levi_civita.transport_products())
    ).astype(np.float64)
    connection = surface_connection(geometry, deviations)
    evidence = connection_holonomy(connection, cycles)

    assert evidence.local_error <= evidence.limit
    assert evidence.generator_error > evidence.limit
    with pytest.raises(SurfaceError, match="not integrable"):
        admit_integrable_connection(connection, cycles)


def test_holonomy_products_preserve_canonical_edge_multiplication_order() -> None:
    geometry = _torus(6, 8)
    levi_civita = levi_civita_connection(geometry)
    deviations = np.linspace(-0.7, 0.9, len(levi_civita.dual_edges()), dtype=np.float64)
    connection = surface_connection(geometry, deviations)
    cycles = integral_dual_cycles(geometry)
    transports = connection.transport_products()
    local_coefficients = _local_dual_cycles(geometry)
    generator_coefficients = cycles.cycle_coefficients().toarray()
    evidence = connection_holonomy(connection, cycles)
    observed = np.asarray(
        (*evidence.local_products, *evidence.generator_products),
        dtype=np.complex128,
    )
    canonical = np.asarray(
        (
            *_represented_cycle_products(transports, local_coefficients),
            *_represented_cycle_products(transports, generator_coefficients),
        ),
        dtype=np.complex128,
    )
    reversed_order = np.asarray(
        (
            *_represented_cycle_products(transports, local_coefficients, reverse=True),
            *_represented_cycle_products(
                transports, generator_coefficients, reverse=True
            ),
        ),
        dtype=np.complex128,
    )

    np.testing.assert_array_equal(observed, canonical)
    assert not np.array_equal(observed, reversed_order)


def test_connection_boundaries_reject_foreign_or_malformed_values() -> None:
    geometry = _tetrahedron()
    connection = _trivial_connection(geometry)
    foreign = _tetrahedron()
    with pytest.raises(SurfaceError, match="same geometry"):
        connection_holonomy(connection, integral_dual_cycles(foreign))
    with pytest.raises(SurfaceError, match="deviation"):
        surface_connection(
            geometry,
            np.zeros(len(connection.dual_edges()) + 1, dtype=np.float64),
        )
    with pytest.raises(SurfaceError, match="adjacent"):
        connection.transport(0, 0)
    with pytest.raises(SurfaceError, match="must be created"):
        from polygeo import IntegrableConnection

        IntegrableConnection()
