# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type

import numpy as np

from polygeo import Complex, Geometry, disk, gaussian_curvature_measure

positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
raw = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
triangle = raw.triangle_manifold()
disk(triangle.with_boundary().connected())
disk(triangle.oriented().with_boundary())
disk(triangle.oriented().connected())
geometry = Geometry.from_positions(raw, positions)
gaussian_curvature_measure(geometry)
