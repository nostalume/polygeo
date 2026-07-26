from typing import Literal, assert_type

import numpy as np
from scipy.sparse import csr_array

from polygeo import (
    BoundaryUnknown,
    CochainSpace,
    Complex,
    ConnectivityUnknown,
    DualCochainSpace,
    FieldSemantics,
    Form,
    Geometry,
    LinearMap,
    OrientationUnknown,
    Simplicial,
    exterior_derivative,
    hodge_star,
)


type Domain = Complex[
    BoundaryUnknown,
    OrientationUnknown,
    ConnectivityUnknown,
    Simplicial,
]


class AlternateSemantics(FieldSemantics):
    pass


complex_ = Complex.from_maximal_simplices(np.array([[0, 1, 2, 3, 4]], dtype=np.int64))
degree_2: Literal[2] = 2
degree_3: Literal[3] = 3
degree_4: Literal[4] = 4
space_2 = complex_.cochain_space(degree_2)
space_3 = complex_.cochain_space(degree_3)
space_4 = complex_.cochain_space(degree_4)
assert_type(space_2, CochainSpace[Domain, Literal[2]])
assert_type(space_3, CochainSpace[Domain, Literal[3]])
assert_type(space_4, CochainSpace[Domain, Literal[4]])

positions = np.vstack(
    [
        np.zeros((1, 4), dtype=np.float64),
        np.eye(4, dtype=np.float64),
    ]
)
geometry = Geometry.from_positions(complex_, positions)
star_3 = hodge_star(geometry, space_3)
assert_type(
    star_3,
    LinearMap[
        CochainSpace[Domain, Literal[3]],
        DualCochainSpace[Domain, Literal[3]],
    ],
)
assert_type(star_3.target, DualCochainSpace[Domain, Literal[3]])
assert_type(star_3.target.primal_degree, Literal[3])
assert_type(star_3.target.degree, int)

derivative = exterior_derivative(space_3, space_4)
assert_type(
    derivative,
    LinearMap[
        CochainSpace[Domain, Literal[3]],
        CochainSpace[Domain, Literal[4]],
    ],
)
assert_type(derivative.matrix(), csr_array)
derivative_2 = exterior_derivative(space_2, space_3)
assert_type(
    derivative.compose(derivative_2),
    LinearMap[
        CochainSpace[Domain, Literal[2]],
        CochainSpace[Domain, Literal[4]],
    ],
)

semantics = AlternateSemantics()
value = space_3.form(np.zeros(space_3.size), semantics)
assert_type(
    derivative.apply(value),
    Form[CochainSpace[Domain, Literal[4]], AlternateSemantics],
)
assert_type(
    star_3.apply(value),
    Form[DualCochainSpace[Domain, Literal[3]], AlternateSemantics],
)


def runtime_derivative(
    degree: int,
) -> LinearMap[
    CochainSpace[Domain, int],
    CochainSpace[Domain, int],
]:
    runtime_source = complex_.cochain_space(degree)
    runtime_target = complex_.cochain_space(degree + 1)
    return exterior_derivative(runtime_source, runtime_target)


assert_type(
    runtime_derivative(2),
    LinearMap[CochainSpace[Domain, int], CochainSpace[Domain, int]],
)


def runtime_hodge(
    degree: int,
) -> LinearMap[
    CochainSpace[Domain, int],
    DualCochainSpace[Domain, int],
]:
    runtime_source = complex_.cochain_space(degree)
    return hodge_star(geometry, runtime_source)


assert_type(
    runtime_hodge(2),
    LinearMap[CochainSpace[Domain, int], DualCochainSpace[Domain, int]],
)

complex_8 = Complex.from_maximal_simplices(np.array([list(range(9))], dtype=np.int64))
degree_7: Literal[7] = 7
degree_8: Literal[8] = 8
space_7 = complex_8.cochain_space(degree_7)
space_8 = complex_8.cochain_space(degree_8)
derivative_7 = exterior_derivative(space_7, space_8)
positions_8 = np.vstack(
    [
        np.zeros((1, 8), dtype=np.float64),
        np.eye(8, dtype=np.float64),
    ]
)
geometry_8 = Geometry.from_positions(complex_8, positions_8)
star_7 = hodge_star(geometry_8, space_7)
assert_type(
    star_7,
    LinearMap[
        CochainSpace[Domain, Literal[7]],
        DualCochainSpace[Domain, Literal[7]],
    ],
)
assert_type(
    derivative_7,
    LinearMap[
        CochainSpace[Domain, Literal[7]],
        CochainSpace[Domain, Literal[8]],
    ],
)
value_7 = space_7.form(np.zeros(space_7.size), semantics)
assert_type(
    derivative_7.apply(value_7),
    Form[CochainSpace[Domain, Literal[8]], AlternateSemantics],
)
