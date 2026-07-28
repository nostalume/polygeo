from __future__ import annotations

import json
import sys

import numpy as np

from polygeo import (
    ORDINARY_FORM,
    CochainSpace,
    Complex,
    Geometry,
    PlotError,
    plot_cochain,
    plot_geometry,
    plot_homology_cycle,
    plot_surface_vectors,
    VertexVectors,
    real_homology_basis,
)


mode = sys.argv[1]
if mode not in {"without-extra", "with-extra"}:
    raise SystemExit("expected without-extra or with-extra")

triangle = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
triangle_geometry = Geometry.from_positions(
    triangle,
    np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 0.8]], dtype=np.float64),
)
form = CochainSpace(triangle, 0).form(
    np.array([-1.0, 0.0, 2.0], dtype=np.float64), ORDINARY_FORM
)

cycle = Complex.from_maximal_simplices(
    np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
)
cycle_geometry = Geometry.from_positions(
    cycle,
    np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    ),
)
basis = real_homology_basis(cycle, 1)
calls = (
    lambda: plot_geometry(triangle_geometry),
    lambda: plot_cochain(triangle_geometry, form),
    lambda: plot_homology_cycle(cycle_geometry, basis, 0),
    lambda: plot_surface_vectors(
        VertexVectors(triangle_geometry, np.ones_like(triangle_geometry.positions))
    ),
)

if mode == "without-extra":
    assert not any(
        name == "plotly" or name.startswith("plotly.") for name in sys.modules
    )
    for call in calls:
        try:
            call()
        except PlotError as error:
            assert str(error) == (
                "plotting requires the optional polygeo[plot] dependency"
            )
        else:
            raise AssertionError("plotting unexpectedly succeeded without the extra")
else:
    for call in calls:
        payload = json.loads(call().to_json())
        assert payload["data"]
