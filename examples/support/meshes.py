"""Deterministic mesh fixtures shared by the executable mathematical studies."""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from polygeo import Complex, Geometry, MeshError


def icosphere(subdivisions: int, radius: float = 1.0):
    """Return an oriented closed Trimesh icosphere as complete PolyGeo geometry."""
    if type(subdivisions) is not int or subdivisions < 0 or subdivisions > 3:
        raise ValueError("icosphere subdivisions must be an integer from zero to three")
    if type(radius) is not float or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("icosphere radius must be finite and positive")
    try:
        import trimesh
    except ImportError as error:
        raise MeshError(
            "study sphere generation requires the optional polygeo[mesh] dependency"
        ) from error
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
    if major_sections < 3 or minor_sections < 3:
        raise ValueError("torus sections must be at least three")
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
    return (
        domain,
        Geometry.from_positions(domain, positions),
        np.tile(v, major_sections),
    )


def annulus(
    radial_rings: int,
    angular_sections: int,
    inner_radius: float = 1.0,
    outer_radius: float = 3.0,
):
    """Return a deterministic positive-Hodge-compatible Delaunay annulus candidate."""
    if radial_rings < 3 or angular_sections < 8:
        raise ValueError("annulus resolution is too small")
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


__all__ = ["annulus", "icosphere", "torus"]
