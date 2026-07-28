# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type

import math
import numpy as np

from polygeo import (
    Complex,
    FrozenFlowEvidence,
    Geometry,
    PositiveHodgeMetric,
    ResidualEvidence,
    mean_curvature_flow_step,
    prepare_direct,
    prepare_least_squares,
    vertex_map,
)


faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
raw = Complex.from_maximal_simplices(faces)
triangle = raw.triangle_manifold()
closed = triangle.without_boundary()
connected = closed.connected()
positions = np.array(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
    dtype=np.float64,
)
geometry = Geometry.from_positions(connected, positions)
metric = PositiveHodgeMetric(geometry)

bounded = triangle.with_boundary().connected()
bounded_metric = PositiveHodgeMetric(Geometry.from_positions(bounded, positions))
mean_curvature_flow_step(bounded_metric, 0.1, prepare_direct)

unconnected_metric = PositiveHodgeMetric(Geometry.from_positions(closed, positions))
mean_curvature_flow_step(unconnected_metric, 0.1, prepare_direct)

cycle_raw = Complex.from_maximal_simplices(
    np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
)
cycle = cycle_raw.codimension_one_regular().without_boundary().connected()
cycle_metric = PositiveHodgeMetric(
    Geometry.from_positions(
        cycle,
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
    )
)
mean_curvature_flow_step(cycle_metric, 0.1, prepare_direct)
mean_curvature_flow_step(metric, "0.1", prepare_direct)
mean_curvature_flow_step(metric, math.pi, prepare_least_squares)
vertex_map(geometry, geometry, "3")
residual = ResidualEvidence(0.0, 0.0, 1.0e-8)
descriptive = FrozenFlowEvidence(0.1, 1.0, 0.5, residual, (residual,))
mean_curvature_flow_step(descriptive, 0.1, prepare_direct)
