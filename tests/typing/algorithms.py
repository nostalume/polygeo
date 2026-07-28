from typing import Literal, assert_type

import numpy as np
from numpy.typing import NDArray

from polygeo import (
    ORDINARY_FORM,
    AssembledSystem,
    BasisCoordinates,
    BoundaryUnknown,
    Certified,
    CochainSpace,
    CodimensionOneRegular,
    Complex,
    Connected,
    ConnectivityUnknown,
    Geometry,
    HodgeComponents,
    HodgeEvidence,
    LinearSolution,
    MeanZeroProblem,
    OrdinaryForm,
    OrientationUnknown,
    PositiveHodgeMetric,
    RealHomologyBasis,
    ResidualEvidence,
    Simplicial,
    WithoutBoundary,
    assemble_poisson,
    hodge_decomposition,
    impose_mean_zero,
    prepare_direct,
    prepare_least_squares,
    real_homology_basis,
)


type Domain = Complex[
    BoundaryUnknown,
    OrientationUnknown,
    ConnectivityUnknown,
    Simplicial,
]

complex_: Domain = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
geometry = Geometry.from_positions(
    complex_, np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
)
metric = PositiveHodgeMetric(geometry)
assert_type(metric, PositiveHodgeMetric[Domain])
assert_type(metric.geometry, Geometry[Domain])
assert_type(metric.weights(0), NDArray[np.float64])
degree_four: Literal[4] = 4
complex_four: Domain = Complex.from_maximal_simplices(
    np.arange(5, dtype=np.int64)[None, :]
)
homology_four = real_homology_basis(complex_four, degree_four)
assert_type(homology_four, RealHomologyBasis[Domain, Literal[4]])
assert_type(
    homology_four.periods(
        complex_four.cochain_space(degree_four).form(
            np.zeros(complex_four.simplex_count(4)), ORDINARY_FORM
        )
    ),
    BasisCoordinates[RealHomologyBasis[Domain, Literal[4]]],
)
runtime_degree: int = complex_.dimension
assert_type(
    real_homology_basis(complex_, runtime_degree), RealHomologyBasis[Domain, int]
)
assert_type(
    hodge_decomposition(
        PositiveHodgeMetric(
            Geometry.from_positions(complex_four, np.eye(5, dtype=np.float64))
        ),
        complex_four.cochain_space(degree_four).form(
            np.zeros(complex_four.simplex_count(4)), ORDINARY_FORM
        ),
        prepare_least_squares,
    ),
    Certified[HodgeComponents[Domain, Literal[4]], HodgeEvidence],
)
degree_seven: Literal[7] = 7
complex_seven: Domain = Complex.from_maximal_simplices(
    np.arange(8, dtype=np.int64)[None, :]
)
assert_type(
    hodge_decomposition(
        PositiveHodgeMetric(
            Geometry.from_positions(complex_seven, np.eye(8, dtype=np.float64))
        ),
        complex_seven.cochain_space(degree_seven).form(
            np.zeros(complex_seven.simplex_count(7)), ORDINARY_FORM
        ),
        prepare_least_squares,
    ),
    Certified[HodgeComponents[Domain, Literal[7]], HodgeEvidence],
)
C0 = complex_.cochain_space(0)
density = C0.form(np.ones(C0.size), ORDINARY_FORM)
assert_type(
    assemble_poisson(metric, density),
    AssembledSystem[
        CochainSpace[Domain, Literal[0]],
        CochainSpace[Domain, Literal[0]],
        OrdinaryForm,
    ],
)

type ClosedDomain = Complex[
    WithoutBoundary,
    OrientationUnknown,
    Connected,
    CodimensionOneRegular,
]
closed_raw = Complex.from_maximal_simplices(
    np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
)
closed: ClosedDomain = (
    closed_raw.codimension_one_regular().without_boundary().connected()
)
closed_geometry = Geometry.from_positions(
    closed,
    np.array(
        [[1.0, 0.0], [-0.5, 0.8660254037844386], [-0.5, -0.8660254037844386]],
        dtype=np.float64,
    ),
)
closed_metric = PositiveHodgeMetric(closed_geometry)
closed_C0 = closed.cochain_space(0)
closed_density = closed_C0.form(np.zeros(closed_C0.size), ORDINARY_FORM)
problem = impose_mean_zero(closed_metric, closed_density)
assert_type(problem, MeanZeroProblem[ClosedDomain])
assert_type(problem.compatibility_evidence, ResidualEvidence)
assert_type(
    problem.solve(prepare_direct),
    LinearSolution[
        CochainSpace[ClosedDomain, Literal[0]],
        CochainSpace[ClosedDomain, Literal[0]],
        OrdinaryForm,
    ],
)
