# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type

import numpy as np

from polygeo import (
    ORDINARY_FORM,
    Complex,
    Geometry,
    HolonomyEvidence,
    admit_integrable_connection,
    integral_dual_cycles,
    integrate_direction_field,
    levi_civita_connection,
    surface_connection,
)

faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
raw = Complex.from_maximal_simplices(faces)
closed = raw.triangle_manifold().oriented().without_boundary().connected()
geometry = Geometry.from_positions(
    closed,
    np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=np.float64,
    ),
)
connection = levi_civita_connection(geometry)
cycles = integral_dual_cycles(geometry)
ordinary_form = closed.cochain_space(1).form(
    np.zeros(closed.simplex_count(1), dtype=np.float64), ORDINARY_FORM
)

surface_connection(geometry, ordinary_form)
integrate_direction_field(connection)
admit_integrable_connection(connection, HolonomyEvidence((), (), 0.0, 0.0, 0.0))
levi_civita_connection(
    Geometry.from_positions(
        raw.triangle_manifold().oriented().with_boundary().connected(),
        geometry.positions,
    )
)
