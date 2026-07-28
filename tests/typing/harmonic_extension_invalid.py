# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type

import numpy as np

from polygeo import (
    ORDINARY_FORM,
    CochainSubspace,
    Complex,
    Geometry,
    PositiveHodgeMetric,
    harmonic_extension,
    prepare_direct,
    prepare_least_squares,
    topological_boundary,
)


raw = Complex.from_maximal_simplices(np.array([[0, 1], [1, 2]], dtype=np.int64))
positions = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
regular = raw.codimension_one_regular()
bounded = regular.with_boundary()
connected = bounded.connected()
parent = connected.cochain_space(0)
boundary = CochainSubspace(
    parent,
    np.flatnonzero(topological_boundary(connected).mask(0)).astype(np.int64),
)
values = boundary.form(np.array([0.0, 2.0]), ORDINARY_FORM)

unconnected_metric = PositiveHodgeMetric(Geometry.from_positions(bounded, positions))
harmonic_extension(unconnected_metric, values, prepare_direct)

unknown_boundary_metric = PositiveHodgeMetric(
    Geometry.from_positions(regular.connected(), positions)
)
harmonic_extension(unknown_boundary_metric, values, prepare_direct)

raw_metric = PositiveHodgeMetric(Geometry.from_positions(raw, positions))
harmonic_extension(raw_metric, values, prepare_direct)

metric = PositiveHodgeMetric(Geometry.from_positions(connected, positions))
harmonic_extension(metric, values, prepare_least_squares)
