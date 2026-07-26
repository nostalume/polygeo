from typing import Literal, assert_type

import numpy as np

from polygeo import (
    CodimensionOneRegular,
    BoundaryUnknown,
    CochainSpace,
    CochainSubspace,
    Complex,
    ConnectivityUnknown,
    FieldSemantics,
    Form,
    LinearMap,
    OrientationUnknown,
    SimplexSubset,
    Simplicial,
    TriangleManifold,
    extend_zero,
    restrict,
    topological_boundary,
)

type Raw = Complex[
    BoundaryUnknown,
    OrientationUnknown,
    ConnectivityUnknown,
    Simplicial,
]
type Regular = Complex[
    BoundaryUnknown,
    OrientationUnknown,
    ConnectivityUnknown,
    CodimensionOneRegular,
]
type Triangle = Complex[
    BoundaryUnknown,
    OrientationUnknown,
    ConnectivityUnknown,
    TriangleManifold,
]
type ParentOne = CochainSpace[Regular, Literal[1]]
type SubspaceOne = CochainSubspace[ParentOne]

raw: Raw = Complex.from_maximal_simplices(
    np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
)
regular = raw.codimension_one_regular()
assert_type(regular, Regular)
assert_type(topological_boundary(regular), SimplexSubset[Regular])

triangle: Triangle = raw.triangle_manifold()
assert_type(topological_boundary(triangle), SimplexSubset[Triangle])

parent = regular.cochain_space(1)
subspace = CochainSubspace(parent, np.array([0, 2, 4], dtype=np.int64))
assert_type(subspace, SubspaceOne)
assert_type(restrict(parent, subspace), LinearMap[ParentOne, SubspaceOne])
assert_type(extend_zero(subspace, parent), LinearMap[SubspaceOne, ParentOne])
assert_type(subspace.complement(), SubspaceOne)


class AlternateSemantics(FieldSemantics):
    pass


alternate = AlternateSemantics()
parent_value = parent.form(np.zeros(parent.size), alternate)
restricted_value = restrict(parent, subspace).apply(parent_value)
assert_type(restricted_value, Form[SubspaceOne, AlternateSemantics])
assert_type(
    extend_zero(subspace, parent).apply(restricted_value),
    Form[ParentOne, AlternateSemantics],
)


class SpecialSubspace(CochainSubspace[CochainSpace[Regular, Literal[1]]]):
    __slots__ = ()


special = SpecialSubspace(parent, np.array([0], dtype=np.int64))
assert_type(special.complement(), SpecialSubspace)


def runtime_subspace(
    degree: int,
) -> CochainSubspace[CochainSpace[Regular, int]]:
    parent = regular.cochain_space(degree)
    return CochainSubspace(parent, np.array([], dtype=np.int64))


def runtime_restriction(
    degree: int,
) -> LinearMap[
    CochainSpace[Regular, int],
    CochainSubspace[CochainSpace[Regular, int]],
]:
    parent = regular.cochain_space(degree)
    subspace = CochainSubspace(parent, np.array([], dtype=np.int64))
    return restrict(parent, subspace)


def runtime_extension(
    degree: int,
) -> LinearMap[
    CochainSubspace[CochainSpace[Regular, int]],
    CochainSpace[Regular, int],
]:
    parent = regular.cochain_space(degree)
    subspace = CochainSubspace(parent, np.array([], dtype=np.int64))
    return extend_zero(subspace, parent)


assert_type(
    runtime_subspace(1),
    CochainSubspace[CochainSpace[Regular, int]],
)
assert_type(
    runtime_restriction(1),
    LinearMap[
        CochainSpace[Regular, int],
        CochainSubspace[CochainSpace[Regular, int]],
    ],
)
assert_type(
    runtime_extension(1),
    LinearMap[
        CochainSubspace[CochainSpace[Regular, int]],
        CochainSpace[Regular, int],
    ],
)
