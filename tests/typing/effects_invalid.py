# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type

from typing import Literal

from polygeo import (
    Binary64Cochain,
    Geometry,
    HomologyGroup,
    PositiveMetric,
    load_surface,
    plot_form,
    plot_geometry,
    plot_homology_cycle,
    plot_surface_vectors,
)

load_surface(3)
plot_geometry(object())


def invalid_effect_values(geometry: Geometry) -> None:
    plot_form(geometry, object())
    plot_homology_cycle(geometry, object(), 0)


plot_surface_vectors(Geometry)


def incompatible_problems(metric: PositiveMetric, source: Binary64Cochain[int]) -> None:
    heat = metric.heat_evolution(source, 0.1)
    poisson = metric.mean_zero_poisson_density(source)
    prepared = heat.prepare()
    prepared.solve(poisson, prepared.workspace_for(heat))


def wrong_harmonic_basis_degree(
    metric: PositiveMetric, group: HomologyGroup[Literal[0]]
) -> None:
    metric.harmonic_one_form_basis(group)
