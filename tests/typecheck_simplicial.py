from typing import Literal, assert_type

import numpy as np

from polygeo import (
    ORDINARY_FORM,
    Closed,
    Complex,
    Connected,
    Form,
    OrdinaryForm,
    Oriented,
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
        Complex[Closed, Oriented, Connected, TriangleManifold],
        Literal[1],
        OrdinaryForm,
    ],
)
