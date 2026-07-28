from typing import assert_type

import numpy as np

from polygeo import (
    Certified,
    Complex,
    Connected,
    DirectionFieldEvidence,
    FaceDirectionField,
    Geometry,
    HolonomyEvidence,
    IntegralDualCycles,
    IntegrableConnection,
    Oriented,
    SurfaceConnection,
    TriangleFrames,
    TriangleManifold,
    WithoutBoundary,
    admit_integrable_connection,
    connection_holonomy,
    integral_dual_cycles,
    integrate_direction_field,
    levi_civita_connection,
    surface_connection,
    triangle_frames,
)


Domain = Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]
raw = Complex.from_maximal_simplices(
    np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
)
domain: Domain = raw.triangle_manifold().oriented().without_boundary().connected()
geometry = Geometry.from_positions(
    domain,
    np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=np.float64,
    ),
)
frames = triangle_frames(geometry)
connection = levi_civita_connection(geometry)
deviations = -np.angle(connection.transport_products()).astype(np.float64)
connection = surface_connection(geometry, deviations)
cycles = integral_dual_cycles(geometry)
holonomy = connection_holonomy(connection, cycles)
capability = admit_integrable_connection(connection, cycles)
field = integrate_direction_field(capability)

assert_type(frames, TriangleFrames[Domain])
assert_type(connection, SurfaceConnection[Domain])
assert_type(cycles, IntegralDualCycles[Domain])
assert_type(holonomy, HolonomyEvidence)
assert_type(capability, IntegrableConnection[Domain])
assert_type(
    field,
    Certified[FaceDirectionField[Domain], DirectionFieldEvidence],
)
