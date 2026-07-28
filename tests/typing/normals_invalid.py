# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type

import numpy as np

from polygeo import (
    Complex,
    Geometry,
    PositiveHodgeMetric,
    face_unit_normals,
    mean_curvature_vectors,
    volume_gradient,
)

positions = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
raw = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
unoriented = raw.triangle_manifold().with_boundary().connected()
oriented_boundary = unoriented.oriented()
face_unit_normals(Geometry.from_positions(unoriented, positions))
volume_gradient(Geometry.from_positions(oriented_boundary, positions))
mean_curvature_vectors(
    PositiveHodgeMetric(Geometry.from_positions(oriented_boundary, positions))
)
