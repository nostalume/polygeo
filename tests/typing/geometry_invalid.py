# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type

import numpy as np
from numpy.typing import NDArray

from polygeo import (
    BoundaryUnknown,
    Complex,
    ConnectivityUnknown,
    Geometry,
    OrientationUnknown,
    TriangleManifold,
)

raw = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
triangle = raw.triangle_manifold()
positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    dtype=np.float64,
)
raw_geometry = Geometry.from_positions(raw, positions)
triangle_geometry = Geometry.from_positions(triangle, positions)


def consume_triangle(
    value: Geometry[
        Complex[
            BoundaryUnknown,
            OrientationUnknown,
            ConnectivityUnknown,
            TriangleManifold,
        ]
    ],
) -> None:
    pass


consume_triangle(raw_geometry)

Geometry.from_positions(
    raw,
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
)


class FakeDomain:
    @property
    def vertex_count(self) -> int:
        return 0

    @property
    def dimension(self) -> int:
        return 0

    def simplex_count(self, degree: int) -> int:
        return 0

    def simplices(self, degree: int) -> NDArray[np.int64]:
        return np.empty((0, degree + 1), dtype=np.int64)


Geometry(FakeDomain(), np.empty((0, 0), dtype=np.float64))
