from pathlib import Path
from typing import assert_type

from plotly.graph_objects import Figure

from polygeo import (
    Binary64Element,
    EntityVectors,
    EuclideanRealization,
    Geometry,
    HomologyGroup,
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
