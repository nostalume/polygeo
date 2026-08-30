from typing import Literal, assert_type

import numpy as np

from polygeo import (
    Binary64Cochain,
    Binary64Element,
    Binary64Space,
    ChainIsomorphism,
    Complex,
    Geometry,
    HalfedgeSurface,
    LinearOperator,
    SurfaceCorrespondence,
)

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
assert_type(geometry.complex, Complex)

space = domain.binary64_cochain_space(0)
selected = domain.binary64_cochain_space(0, indices=np.array([0, 2], dtype=np.int64))
form = space.admit_numpy(np.zeros(space.size, dtype=np.float64))
assert_type(space, Binary64Space[Literal["cochain"], int])
assert_type(selected, Binary64Space[Literal["cochain"], int])
assert_type(form, Binary64Element[Literal["cochain"], int])
assert_type(form, Binary64Cochain[int])
assert_type(
    space.identity(),
    LinearOperator[Literal["cochain"], int, Literal["cochain"], int],
)

surface_domain = (
    Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    .triangle_manifold()
    .oriented()
)
surface, correspondence = HalfedgeSurface.from_complex(surface_domain)
assert_type(surface, HalfedgeSurface)
assert_type(correspondence, SurfaceCorrespondence)
assert_type(correspondence.chain_isomorphism(), ChainIsomorphism[int])
