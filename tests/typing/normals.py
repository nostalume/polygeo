from typing import assert_type

import numpy as np

from polygeo import (
    Complex,
    Connected,
    FaceVectors,
    Geometry,
    Oriented,
    PositiveHodgeMetric,
    TriangleManifold,
    VertexVectors,
    WithoutBoundary,
    face_unit_normals,
    mean_curvature_vectors,
    sphere_inscribed_vertex_normals,
    surface_area_gradient,
    tip_angle_vertex_normals,
    uniform_vertex_normals,
    volume_gradient,
)


type Domain = Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
raw = Complex.from_maximal_simplices(
    np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
)
domain: Domain = raw.triangle_manifold().oriented().without_boundary().connected()
geometry = Geometry.from_positions(
    domain,
    np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=np.float64,
    ),
)
assert_type(face_unit_normals(geometry), FaceVectors[Domain])
assert_type(surface_area_gradient(geometry), VertexVectors[Domain])
assert_type(volume_gradient(geometry), VertexVectors[Domain])
assert_type(uniform_vertex_normals(geometry), VertexVectors[Domain])
assert_type(tip_angle_vertex_normals(geometry), VertexVectors[Domain])
assert_type(sphere_inscribed_vertex_normals(geometry), VertexVectors[Domain])
assert_type(
    mean_curvature_vectors(PositiveHodgeMetric(geometry)), VertexVectors[Domain]
)
