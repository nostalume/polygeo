from pathlib import Path
from typing import Literal, assert_type

from plotly.graph_objects import Figure

from polygeo.chain import HomologyGroup, IntegralCochain
from polygeo.field import (
    Direction,
    DualCycles,
    HarmonicBasis,
    IntegrableConnection,
    Singularities,
)
from polygeo.form import Cochain, Element
from polygeo.geometry import (
    ConformalMap,
    FaceField,
    FlowStep,
    Geometry,
    Metric,
    TriangleSurface,
    VectorField,
    VertexField,
)
from polygeo.mesh import load_surface
from polygeo.plot import direction as plot_direction, form as plot_form
from polygeo.plot import geometry as plot_geometry
from polygeo.plot import homology_cycle, vectors
from polygeo.solve import (
    Executor,
    HeatResult,
    Policy,
    Prepared,
    Problem,
    StorageLimit,
    WorkLimit,
)


def accepted(
    path: Path,
    geometry: Geometry,
    field: VectorField[int],
    form: Element,
    group: HomologyGroup,
) -> None:
    assert_type(load_surface(path), Geometry)
    assert_type(plot_geometry(geometry), Figure)
    assert_type(plot_form(geometry, form), Figure)
    assert_type(homology_cycle(geometry, group, 0), Figure)
    assert_type(vectors(field), Figure)


def numerical_effects(
    metric: Metric,
    source: Cochain[int],
    group: HomologyGroup[Literal[1]],
) -> None:
    problem = metric.heat_evolution(source, 0.1)
    assert_type(problem, Problem[HeatResult])
    policy = Policy(
        executor=Executor.sequential(),
        storage=StorageLimit(1024, 4096),
        work=WorkLimit(1_000_000),
    )
    assert_type(policy.executor, Executor)
    assert_type(policy.storage, StorageLimit)
    assert_type(policy.work, WorkLimit)
    prepared = problem.prepare(policy=policy)
    assert_type(prepared, Prepared[HeatResult])
    workspace = prepared.workspace_for(problem)
    assert_type(prepared.solve(problem, workspace), HeatResult)
    assert_type(metric.frozen_mean_curvature_flow(0.1), FlowStep)
    assert_type(metric.harmonic_basis(group), HarmonicBasis)


def surface_effects(
    surface: TriangleSurface,
    metric: Metric,
    harmonic: HarmonicBasis,
    cycles: DualCycles,
    charges: IntegralCochain[Literal[0]],
) -> None:
    assert_type(
        surface.conformal_map((0, 1)),
        ConformalMap,
    )
    direction = surface.direction_field(2, metric, harmonic, cycles, charges, (), 0.0)
    assert_type(direction, Direction)
    assert_type(plot_direction(direction), Figure)
    assert_type(direction.connection, IntegrableConnection)
    assert_type(direction.symmetry_order, int)
    assert_type(direction.ambient_branch_numpy_copy(0), FaceField)
    assert_type(surface.face_unit_normals(), FaceField)
    assert_type(surface.uniform_vertex_normals(), VertexField)
    singularity_evidence = direction.singularities()
    assert_type(singularity_evidence, Singularities)
    assert_type(singularity_evidence.symmetry_order, int)
    assert_type(surface.boundary_direction(2, metric, 0.0), Direction)
