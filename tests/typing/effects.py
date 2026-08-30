from pathlib import Path
from typing import Literal, assert_type

from plotly.graph_objects import Figure

from polygeo import (
    Binary64Cochain,
    Binary64Element,
    EntityVectors,
    EuclideanRealization,
    FlowStep,
    Geometry,
    HarmonicOneFormBasis,
    HeatSolution,
    HomologyGroup,
    LeastSquaresConformalMapSolution,
    PositiveMetric,
    PreparedProblem,
    Problem,
    TriangleSurface,
    load_surface,
    plot_form,
    plot_geometry,
    plot_homology_cycle,
    plot_surface_vectors,
)


def accepted(
    path: Path,
    geometry: Geometry,
    field: EntityVectors,
    form: Binary64Element,
    group: HomologyGroup,
) -> None:
    assert_type(load_surface(path), EuclideanRealization)
    assert_type(plot_geometry(geometry), Figure)
    assert_type(plot_form(geometry, form), Figure)
    assert_type(plot_homology_cycle(geometry, group, 0), Figure)
    assert_type(plot_surface_vectors(field), Figure)


def numerical_effects(
    metric: PositiveMetric,
    source: Binary64Cochain[int],
    group: HomologyGroup[Literal[1]],
) -> None:
    problem = metric.heat_evolution(source, 0.1)
    assert_type(problem, Problem[HeatSolution])
    prepared = problem.prepare()
    assert_type(prepared, PreparedProblem[HeatSolution])
    workspace = prepared.workspace_for(problem)
    assert_type(prepared.solve(problem, workspace), HeatSolution)
    assert_type(metric.frozen_mean_curvature_flow(0.1), FlowStep)
    assert_type(metric.harmonic_one_form_basis(group), HarmonicOneFormBasis)


def surface_effects(surface: TriangleSurface) -> None:
    assert_type(
        surface.least_squares_conformal_map((0, 1)),
        LeastSquaresConformalMapSolution,
    )
