from typing import Literal, assert_type

import numpy as np

from polygeo import (
    ORDINARY_FORM,
    BoundaryUnknown,
    Closed,
    CochainSpace,
    Complex,
    Connected,
    ConnectivityUnknown,
    Form,
    OrdinaryForm,
    OrientationUnknown,
    Oriented,
    Simplicial,
    TriangleManifold,
)

raw = Complex.from_maximal_simplices(
    np.array(
        [
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )
)
domain = raw.triangle_manifold().oriented().closed().connected()
assert_type(
    domain,
    Complex[Closed, Oriented, Connected, TriangleManifold],
)

space = domain.cochain_space(1)
form = space.form(np.zeros(space.size), ORDINARY_FORM)
assert_type(
    form,
    Form[
        CochainSpace[
            Complex[Closed, Oriented, Connected, TriangleManifold],
            Literal[1],
        ],
        OrdinaryForm,
    ],
)

simplex_4 = Complex.from_maximal_simplices(np.array([[0, 1, 2, 3, 4]], dtype=np.int64))
degree_3: Literal[3] = 3
space_3 = simplex_4.cochain_space(degree_3)
assert_type(
    simplex_4.cochain_space(0),
    CochainSpace[
        Complex[
            BoundaryUnknown,
            OrientationUnknown,
            ConnectivityUnknown,
            Simplicial,
        ],
        Literal[0],
    ],
)
assert_type(
    simplex_4.cochain_space(2),
    CochainSpace[
        Complex[
            BoundaryUnknown,
            OrientationUnknown,
            ConnectivityUnknown,
            Simplicial,
        ],
        Literal[2],
    ],
)
assert_type(
    space_3,
    CochainSpace[
        Complex[
            BoundaryUnknown,
            OrientationUnknown,
            ConnectivityUnknown,
            Simplicial,
        ],
        Literal[3],
    ],
)

simplex_8 = Complex.from_maximal_simplices(np.array([list(range(9))], dtype=np.int64))
degree_7: Literal[7] = 7
degree_8: Literal[8] = 8
space_7 = simplex_8.cochain_space(degree_7)
space_8 = simplex_8.cochain_space(degree_8)
assert_type(
    space_7,
    CochainSpace[
        Complex[
            BoundaryUnknown,
            OrientationUnknown,
            ConnectivityUnknown,
            Simplicial,
        ],
        Literal[7],
    ],
)
assert_type(
    space_8,
    CochainSpace[
        Complex[
            BoundaryUnknown,
            OrientationUnknown,
            ConnectivityUnknown,
            Simplicial,
        ],
        Literal[8],
    ],
)
