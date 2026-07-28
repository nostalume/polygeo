from typing import assert_type

import numpy as np
from plotly.graph_objects import Figure

from polygeo import ORDINARY_FORM, CochainSpace, Complex, Geometry, real_homology_basis
from polygeo.plotting import plot_cochain, plot_geometry, plot_homology_cycle


complex_ = Complex.from_maximal_simplices(
    np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
)
geometry = Geometry.from_positions(
    complex_,
    np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64),
)
form = CochainSpace(complex_, 0).form(np.array([1.0, 2.0, 3.0]), ORDINARY_FORM)
basis = real_homology_basis(complex_, 1)

assert_type(plot_geometry(geometry), Figure)
assert_type(plot_cochain(geometry, form), Figure)
assert_type(plot_homology_cycle(geometry, basis, 0), Figure)
