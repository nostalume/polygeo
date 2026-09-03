# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type

from typing import Literal

from polygeo.chain import HomologyGroup
from polygeo.form import Cochain
from polygeo.geometry import Geometry, Metric
from polygeo.mesh import load_surface
from polygeo.plot import (
    direction,
    form,
    geometry as plot_geometry,
    homology_cycle,
    vectors,
)

load_surface(3)
plot_geometry(object())


def invalid_effect_values(geometry: Geometry) -> None:
    form(geometry, object())
    homology_cycle(geometry, object(), 0)


vectors(Geometry)
direction(Geometry)


def incompatible_problems(metric: Metric, source: Cochain[int]) -> None:
    heat = metric.heat_evolution(source, 0.1)
    poisson = metric.mean_zero_poisson_density(source)
    prepared = heat.prepare()
    prepared.solve(poisson, prepared.workspace_for(heat))


def wrong_harmonic_basis_degree(
    metric: Metric, group: HomologyGroup[Literal[0]]
) -> None:
    metric.harmonic_basis(group)
