from __future__ import annotations

from typing import Literal

import numpy as np

from polygeo import (
    ORDINARY_FORM,
    CochainSubspace,
    Complex,
    Geometry,
    PositiveHodgeMetric,
    disk,
    gaussian_curvature_measure,
    harmonic_extension,
    hodge_decomposition,
    mean_curvature_flow_step,
    prepare_direct,
    prepare_least_squares,
    real_homology_basis,
    topological_boundary,
)


complex_ = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
geometry = Geometry.from_positions(
    complex_,
    np.array([[0.0, 0.0], [1.0, 0.0], [0.25, 1.0]], dtype=np.float64),
)
metric = PositiveHodgeMetric(geometry)
degree: Literal[1] = 1
space = complex_.cochain_space(degree)
form = space.form(np.arange(space.size, dtype=np.float64), ORDINARY_FORM)

homology = real_homology_basis(complex_, degree)
assert homology.cycle_coefficients().shape == (space.size, homology.dimension)
assert homology.periods(form).basis is homology

result = hodge_decomposition(metric, form, prepare_least_squares)
np.testing.assert_allclose(
    result.output.exact.coefficients()
    + result.output.coexact.coefficients()
    + result.output.harmonic.coefficients(),
    form.coefficients(),
)

surface_raw = Complex.from_maximal_simplices(
    np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
)
surface = surface_raw.triangle_manifold().without_boundary().connected()
surface_geometry = Geometry.from_positions(
    surface,
    np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=np.float64,
    ),
)
assert gaussian_curvature_measure(surface_geometry).space.complex is surface
flow = mean_curvature_flow_step(
    PositiveHodgeMetric(surface_geometry), 0.1, prepare_direct
)
assert flow.output.source is surface_geometry
assert flow.output.target.complex is surface

bounded = complex_.triangle_manifold().oriented().with_boundary().connected()
assert disk(bounded).complex is bounded
bounded_geometry = Geometry.from_positions(
    bounded,
    np.array([[0.0, 0.0], [1.0, 0.0], [0.25, 1.0]], dtype=np.float64),
)
boundary = CochainSubspace(
    bounded.cochain_space(0),
    np.flatnonzero(topological_boundary(bounded).mask(0)).astype(np.int64),
)
boundary_values = boundary.form(np.array([0.0, 1.0, 2.0]), ORDINARY_FORM)
extension = harmonic_extension(
    PositiveHodgeMetric(bounded_geometry), boundary_values, prepare_direct
)
np.testing.assert_array_equal(
    extension.form.coefficients(), boundary_values.coefficients()
)
