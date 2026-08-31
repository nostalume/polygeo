from pathlib import Path
from typing import Literal, assert_type

from plotly.graph_objects import Figure

from polygeo import (
    Binary64Cochain,
    Binary64Element,
    DirectionFieldSingularities,
    EntityVectors,
    EuclideanRealization,
    FaceDirectionField,
    FlowStep,
    Geometry,
    HarmonicOneFormBasis,
    HeatSolution,
    HomologyGroup,
    IntegralCochain,
    IntegralDualCycleBasis,
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


def surface_effects(
    surface: TriangleSurface,
    metric: PositiveMetric,
    harmonic: HarmonicOneFormBasis,
    cycles: IntegralDualCycleBasis,
    charges: IntegralCochain[Literal[0]],
) -> None:
    assert_type(
        surface.least_squares_conformal_map((0, 1)),
        LeastSquaresConformalMapSolution,
    )
    direction = surface.minimum_energy_direction_field(
        2, metric, harmonic, cycles, charges, (), 0.0
    )
    assert_type(direction, FaceDirectionField)
    assert_type(direction.symmetry_order, int)
    assert_type(direction.ambient_vector_branch_numpy_copy(0), EntityVectors)
    singularity_evidence = direction.singularities()
    assert_type(singularity_evidence, DirectionFieldSingularities)
    assert_type(singularity_evidence.symmetry_order, int)
    assert_type(
        surface.boundary_aligned_direction_field(2, metric, 0.0), FaceDirectionField
    )
