from typing import assert_type

import numpy as np

from polygeo import (
    BoundaryUnknown,
    Complex,
    ConnectivityUnknown,
    Geometry,
    OrientationUnknown,
    Simplicial,
)

complex_ = Complex.from_maximal_simplices(np.array([[0, 1, 2, 3]], dtype=np.int64))

type Domain = Complex[
    BoundaryUnknown,
    OrientationUnknown,
    ConnectivityUnknown,
    Simplicial,
]

positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
geometry = Geometry.from_positions(complex_, positions)
direct_geometry = Geometry(complex_, positions)
assert_type(
    geometry,
    Geometry[Domain],
)
assert_type(direct_geometry, Geometry[Domain])
assert_type(geometry.complex, Domain)
