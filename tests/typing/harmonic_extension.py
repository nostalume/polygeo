from typing import Literal, assert_type

import numpy as np

from polygeo import (
    ORDINARY_FORM,
    CochainSpace,
    CochainSubspace,
    CodimensionOneRegular,
    Complex,
    Connected,
    Geometry,
    LinearSolution,
    OrdinaryForm,
    OrientationUnknown,
    PositiveHodgeMetric,
    WithBoundary,
    harmonic_extension,
    prepare_direct,
    topological_boundary,
)


type Domain = Complex[
    WithBoundary,
    OrientationUnknown,
    Connected,
    CodimensionOneRegular,
]
raw = Complex.from_maximal_simplices(np.array([[0, 1], [1, 2]], dtype=np.int64))
domain: Domain = raw.codimension_one_regular().with_boundary().connected()
geometry = Geometry.from_positions(
    domain, np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
)
metric = PositiveHodgeMetric(geometry)
parent = domain.cochain_space(0)
indices = np.flatnonzero(topological_boundary(domain).mask(0)).astype(np.int64)
boundary = CochainSubspace(parent, indices)
values = boundary.form(np.array([0.0, 2.0]), ORDINARY_FORM)
assert_type(
    harmonic_extension(metric, values, prepare_direct),
    LinearSolution[
        CochainSpace[Domain, Literal[0]],
        CochainSubspace[CochainSpace[Domain, Literal[0]]],
        OrdinaryForm,
    ],
)
