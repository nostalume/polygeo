# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type

import numpy as np
from numpy.typing import NDArray

from polygeo import Complex, Geometry

domain = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    dtype=np.float64,
)

Geometry.from_positions(
    domain,
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
)
domain.binary64_cochain_space(0).admit_numpy([0.0, 1.0, 2.0])


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


Geometry(FakeDomain(), positions)
