# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type

import numpy as np

from polygeo import (
    ORDINARY_FORM,
    Complex,
    FieldSemantics,
    Geometry,
    PositiveHodgeMetric,
    ResidualEvidence,
    assemble_poisson,
    hodge_decomposition,
    impose_mean_zero,
    prepare_direct,
    prepare_least_squares,
    real_homology_basis,
)


class AlternateSemantics(FieldSemantics):
    pass


complex_ = Complex.from_maximal_simplices(np.array([[0, 1]], dtype=np.int64))
PositiveHodgeMetric(complex_)
geometry = Geometry.from_positions(complex_, np.array([[0.0], [1.0]], dtype=np.float64))
metric = PositiveHodgeMetric(geometry)
edge_density = complex_.cochain_space(1).form(np.ones(1), ORDINARY_FORM)
assemble_poisson(metric, edge_density)
vertex_density = complex_.cochain_space(0).form(np.zeros(2), ORDINARY_FORM)
assemble_poisson(ResidualEvidence(0.0, 0.0, 1.0), vertex_density)
bounded = complex_.codimension_one_regular().with_boundary().connected()
bounded_geometry = Geometry.from_positions(
    bounded, np.array([[0.0], [1.0]], dtype=np.float64)
)
bounded_metric = PositiveHodgeMetric(bounded_geometry)
bounded_density = bounded.cochain_space(0).form(np.zeros(2), ORDINARY_FORM)
impose_mean_zero(bounded_metric, bounded_density)
real_homology_basis(metric, 0)
real_homology_basis(complex_, "1")
homology = real_homology_basis(complex_, 1)
alternate_edge = complex_.cochain_space(1).form(np.ones(1), AlternateSemantics())
homology.periods(alternate_edge)
hodge_decomposition(complex_, vertex_density, prepare_least_squares)
hodge_decomposition(metric, alternate_edge, prepare_least_squares)
hodge_decomposition(metric, vertex_density, prepare_direct)
