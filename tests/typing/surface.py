from typing import Literal, assert_type

import numpy as np

from polygeo import (
    CochainSpace,
    Complex,
    Connected,
    Disk,
    Form,
    Geometry,
    OrdinaryForm,
    OrientationUnknown,
    Oriented,
    TriangleManifold,
    WithBoundary,
    disk,
    gaussian_curvature_measure,
)


type Domain = Complex[WithBoundary, Oriented, Connected, TriangleManifold]
raw = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
domain: Domain = raw.triangle_manifold().oriented().with_boundary().connected()
evidence = disk(domain)
assert_type(evidence, Disk[Domain])
assert_type(evidence.complex, Domain)

unoriented = raw.triangle_manifold().with_boundary().connected()
geometry = Geometry.from_positions(
    unoriented,
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
)
assert_type(
    gaussian_curvature_measure(geometry),
    Form[
        CochainSpace[
            Complex[
                WithBoundary,
                OrientationUnknown,
                Connected,
                TriangleManifold,
            ],
            Literal[0],
        ],
        OrdinaryForm,
    ],
)
