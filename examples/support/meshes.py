"""Deterministic mesh fixtures shared by the executable mathematical studies."""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from polygeo import Complex, Geometry


def icosphere(subdivisions: int, radius: float = 1.0):
    """Return an oriented closed Trimesh icosphere as complete PolyGeo geometry."""
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    domain = (
        Complex.from_maximal_simplices(np.asarray(mesh.faces, dtype=np.int64))
        .triangle_manifold()
        .oriented()
        .without_boundary()
        .connected()
    )
    geometry = Geometry.from_positions(
        domain, np.asarray(mesh.vertices, dtype=np.float64)
    )
    return domain, geometry


def torus(
    major_sections: int,
    minor_sections: int,
    major_radius: float = 2.0,
    minor_radius: float = 0.7,
):
    """Return a consistent-diagonal analytic torus and its minor-angle coordinates."""
    u = 2.0 * np.pi * np.arange(major_sections) / major_sections
    v = 2.0 * np.pi * np.arange(minor_sections) / minor_sections
    positions = np.array(
        [
            [
                (major_radius + minor_radius * np.cos(y)) * np.cos(x),
                (major_radius + minor_radius * np.cos(y)) * np.sin(x),
                minor_radius * np.sin(y),
            ]
            for x in u
            for y in v
        ],
        dtype=np.float64,
    )

    def index(i: int, j: int) -> int:
        return (i % major_sections) * minor_sections + (j % minor_sections)

    faces: list[tuple[int, int, int]] = []
    for i in range(major_sections):
        for j in range(minor_sections):
            a, b = index(i, j), index(i + 1, j)
            c, d = index(i + 1, j + 1), index(i, j + 1)
            faces.extend(((a, b, c), (a, c, d)))
    domain = (
        Complex.from_maximal_simplices(np.asarray(faces, dtype=np.int64))
        .triangle_manifold()
        .oriented()
        .without_boundary()
        .connected()
    )
    return domain, Geometry.from_positions(domain, positions)


def annulus(
    radial_rings: int,
    angular_sections: int,
    inner_radius: float = 1.0,
    outer_radius: float = 3.0,
):
    """Return a deterministic positive-Hodge-compatible Delaunay annulus candidate."""
    points: list[np.ndarray] = []
    for ring, radius in enumerate(
        np.linspace(inner_radius, outer_radius, radial_rings)
    ):
        angles = (
            2.0
            * np.pi
            * (np.arange(angular_sections) + 0.173 * ring)
            / angular_sections
        )
        points.append(
            np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
        )
    positions = np.asarray(np.vstack(points), dtype=np.float64)
    candidates = Delaunay(positions).simplices
    centroids = positions[candidates].mean(axis=1)
    faces = candidates[np.linalg.norm(centroids, axis=1) > inner_radius]
    domain = (
        Complex.from_maximal_simplices(np.asarray(faces, dtype=np.int64))
        .triangle_manifold()
        .oriented()
        .with_boundary()
        .connected()
    )
    return domain, Geometry.from_positions(domain, positions)


def disk(radial_rings: int, angular_sections: int):
    """Return a deterministic convex Delaunay disk with a positive Hodge metric."""
    points = [np.array([[0.0, 0.0]], dtype=np.float64)]
    for ring in range(1, radial_rings + 1):
        radius = ring / radial_rings
        angles = (
            2.0
            * np.pi
            * (np.arange(angular_sections) + 0.173 * ring)
            / angular_sections
        )
        points.append(
            np.column_stack(
                (1.4 * radius * np.cos(angles), 0.8 * radius * np.sin(angles))
            )
        )
    positions = np.vstack(points)
    faces = np.asarray(Delaunay(positions).simplices, dtype=np.int64)
    domain = (
        Complex.from_maximal_simplices(faces)
        .triangle_manifold()
        .oriented()
        .with_boundary()
        .connected()
    )
    return domain, Geometry.from_positions(domain, positions)
