"""Operational acceptance for an isolated installed distribution."""

from __future__ import annotations

from importlib.metadata import metadata, version
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np

import polygeo
from polygeo import _polygeo_native
from polygeo import (
    Complex,
    FlowStep,
    Geometry,
    HalfedgeSurface,
    HeatSolution,
    LeastSquaresConformalMapSolution,
    MeshError,
    PlotError,
    PoissonSolution,
    TriangleSurface,
    load_surface,
    plot_form,
    plot_geometry,
    plot_homology_cycle,
    plot_surface_vectors,
    analyze_integral_homology,
)


def _expect_optional_error(
    call: Callable[[], object], expected: type[Exception]
) -> None:
    try:
        call()
    except expected:
        return
    raise AssertionError(f"{call!r} unexpectedly succeeded without its extra")


mode = sys.argv[1] if len(sys.argv) == 2 else ""
if mode not in {"base", "extras"}:
    raise SystemExit("expected base or extras")

distribution = metadata("polygeo")
assert version("polygeo") == "0.1.0"
assert distribution["License-Expression"] == "MIT"
assert Complex is _polygeo_native.Complex
assert HalfedgeSurface is _polygeo_native.HalfedgeSurface
assert "_native" not in polygeo.__all__

complex_ = (
    Complex.from_maximal_simplices(
        np.array(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            dtype=np.int64,
        )
    )
    .triangle_manifold()
    .without_boundary()
    .connected()
)
exact = complex_.chain_complex()
chain = exact[1].element({0: 1 << 130})
assert exact.boundary(1).apply(chain).to_python_copy()[0]
homology = analyze_integral_homology(exact, [0, 1])
assert homology[0].free_rank == 1

geometry = Geometry.from_positions(
    complex_,
    np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    ),
)
space = complex_.binary64_cochain_space(0)
metric = geometry.positive_metric()
heat = metric.heat_evolution(space.admit_numpy(np.array([1.0, 0.0, 0.0, 0.0])), 0.1)
heat_prepared = heat.prepare()
heat_solution = heat_prepared.solve(heat, heat_prepared.workspace_for(heat))
assert isinstance(heat_solution, HeatSolution)
assert heat_solution.residual_bound <= 1.0e-10
flow_step = metric.frozen_mean_curvature_flow(0.1)
assert isinstance(flow_step, FlowStep)
assert flow_step.residual_bound <= 1.0e-10
weights = metric.hodge_coefficients_numpy_copy(0)
density = space.admit_numpy(np.array([weights[1], -weights[0], 0.0, 0.0]))
problem = metric.mean_zero_poisson_density(density)
prepared = problem.prepare()
solution = prepared.solve(problem, prepared.workspace_for(problem))
assert isinstance(solution, PoissonSolution)
assert solution.potential.space.same_space(space)

surface = TriangleSurface.admit(geometry)
field = surface.face_unit_normals()
assert field.vectors_numpy_copy().shape == (surface.face_count, 3)
gradient = surface.gradient(space.admit_numpy(geometry.positions_numpy_copy()[:, 0]))
load = surface.divergence(gradient)
assert gradient.vectors_numpy_copy().shape == (surface.face_count, 3)
assert load.coefficients_numpy_copy().shape == (4,)
assert load.space.variance == "chain"
load_problem = metric.mean_zero_poisson_load(-load)
load_prepared = load_problem.prepare()
load_solution = load_prepared.solve(
    load_problem, load_prepared.workspace_for(load_problem)
)
assert isinstance(load_solution, PoissonSolution)
oriented_triangle = (
    Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    .triangle_manifold()
    .oriented()
)
np.testing.assert_array_equal(
    oriented_triangle.disk_boundary_vertices_numpy_copy(), [0, 1, 2]
)
triangle_geometry = Geometry.from_positions(
    oriented_triangle,
    np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.25, 1.0, 0.5]]),
)
conformal_map = TriangleSurface.admit(triangle_geometry).least_squares_conformal_map(
    (0, 1)
)
assert isinstance(conformal_map, LeastSquaresConformalMapSolution)
np.testing.assert_array_equal(
    conformal_map.realization.positions_numpy_copy()[[0, 1]],
    [[0.0, 0.0], [1.0, 0.0]],
)
assert conformal_map.required_rank == conformal_map.observed_rank == 2
assert conformal_map.minimum_normalized_signed_twice_area > 0.0
converted, correspondence = HalfedgeSurface.from_complex(oriented_triangle)
transport = correspondence.chain_isomorphism()
source_chain = transport.source[1].element({0: 1})
assert (
    transport.inverse(1)
    .apply(transport.forward(1).apply(source_chain))
    .to_python_copy()
    == source_chain.to_python_copy()
)
assert converted.to_complex()[0].simplex_count(2) == 1

with TemporaryDirectory() as directory:
    mesh_path = Path(directory) / "triangle.obj"
    mesh_path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    plot_calls = (
        lambda: plot_geometry(geometry),
        lambda: plot_form(geometry, density),
        lambda: plot_homology_cycle(geometry, homology[0], 0),
        lambda: plot_surface_vectors(field),
    )
    if mode == "base":
        assert importlib.util.find_spec("trimesh") is None
        assert importlib.util.find_spec("plotly") is None
        _expect_optional_error(lambda: load_surface(mesh_path), MeshError)
        for plot_call in plot_calls:
            _expect_optional_error(plot_call, PlotError)
    else:
        assert load_surface(mesh_path).complex.simplex_count(2) == 1
        for plot_call in plot_calls:
            assert json.loads(plot_call().to_json())["data"]
