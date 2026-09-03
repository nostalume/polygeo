from typing import Literal, assert_type

import numpy as np

from polygeo.chain import ChainIsomorphism
from polygeo.form import Cochain, Element, Operator, Space
from polygeo.geometry import Geometry
from polygeo.topology import Complex, HalfedgeSurface

domain = Complex.from_maximal_simplices(np.array([[0, 1, 2, 3]], dtype=np.int64))
positions = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

geometry = Geometry.from_positions(domain, positions)
assert_type(geometry, Geometry[Complex])
assert_type(geometry.topology, Complex)

space = domain.binary64_cochain_space(0)
selected = domain.binary64_cochain_space(0, indices=np.array([0, 2], dtype=np.int64))
form = space.admit_numpy(np.zeros(space.size, dtype=np.float64))
assert_type(space, Space[Literal["cochain"], int])
assert_type(selected, Space[Literal["cochain"], int])
assert_type(form, Element[Literal["cochain"], int])
assert_type(form, Cochain[int])
assert_type(
    space.identity(),
    Operator[Literal["cochain"], int, Literal["cochain"], int],
)

surface_domain = (
    Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    .triangle_manifold()
    .oriented()
)
surface, correspondence = HalfedgeSurface.from_complex(surface_domain)
assert_type(surface, HalfedgeSurface)
assert_type(correspondence, ChainIsomorphism[int])
