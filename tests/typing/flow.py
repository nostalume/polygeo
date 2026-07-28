from typing import Literal, assert_type

import numpy as np

from polygeo import (
    Certified,
    BoundaryUnknown,
    Complex,
    Connected,
    FrozenFlowEvidence,
    Geometry,
    ConnectivityUnknown,
    OrientationUnknown,
    PositiveHodgeMetric,
    Simplicial,
    TriangleManifold,
    VertexMap,
    WithoutBoundary,
    mean_curvature_flow_step,
    prepare_direct,
    vertex_map,
)


type Surface = Complex[
    WithoutBoundary,
    OrientationUnknown,
    Connected,
    TriangleManifold,
]
faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
surface: Surface = (
    Complex.from_maximal_simplices(faces)
    .triangle_manifold()
    .without_boundary()
    .connected()
)
geometry = Geometry.from_positions(
    surface,
    np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=np.float64,
    ),
)
runtime_dimension: int = geometry.ambient_dimension
assert_type(vertex_map(geometry, geometry, runtime_dimension), VertexMap[Surface, int])
assert_type(
    mean_curvature_flow_step(PositiveHodgeMetric(geometry), 0.1, prepare_direct),
    Certified[VertexMap[Surface, int], FrozenFlowEvidence],
)

type Raw = Complex[
    BoundaryUnknown,
    OrientationUnknown,
    ConnectivityUnknown,
    Simplicial,
]
triangle: Raw = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
triangle_geometry = Geometry.from_positions(
    triangle,
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
)
assert_type(
    vertex_map(triangle_geometry, triangle_geometry, 2), VertexMap[Raw, Literal[2]]
)
