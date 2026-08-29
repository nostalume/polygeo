"""Optional Plotly output effects over explicit native snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._polygeo_native import (
    Binary64Element,
    Complex,
    EntityVectors,
    EuclideanRealization,
    HomologyGroup,
)

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

type PlotAxes = tuple[int, int] | tuple[int, int, int]
type FloatArray = NDArray[np.float64]


class PlotError(ValueError):
    """Invalid plot relation or unavailable optional dependency."""


def _project(positions: FloatArray, axes: PlotAxes | None) -> FloatArray:
    if axes is not None:
        if len(axes) not in (2, 3) or len(set(axes)) != len(axes):
            raise PlotError("projection axes must be distinct")
        if any(axis < 0 or axis >= positions.shape[1] for axis in axes):
            raise PlotError("projection axis is outside the ambient dimension")
        return positions[:, axes]
    if positions.shape[1] == 1:
        return np.column_stack((positions[:, 0], np.zeros(len(positions))))
    if positions.shape[1] in (2, 3):
        return positions
    raise PlotError("an explicit projection is required above ambient dimension three")


def _snapshot(
    geometry: EuclideanRealization, axes: PlotAxes | None
) -> tuple[Complex, FloatArray]:
    topology = geometry.complex
    if topology.dimension > 2:
        raise PlotError("plotting supports intrinsic dimensions zero through two")
    return topology, _project(geometry.positions_numpy_copy(), axes)


def _coordinates(values: FloatArray) -> dict[str, object]:
    return {
        "x": values[:, 0],
        "y": values[:, 1],
        **({"z": values[:, 2]} if values.shape[1] == 3 else {}),
    }


def _scatter(
    values: FloatArray,
    *,
    mode: str,
    name: str,
    **properties: object,
) -> dict[str, object]:
    return {
        "type": "scatter" if values.shape[1] == 2 else "scatter3d",
        **_coordinates(values),
        "mode": mode,
        "name": name,
        **properties,
    }


def _geometry_traces(
    topology: Complex, points: FloatArray
) -> tuple[dict[str, object], ...]:
    if topology.dimension == 2 and points.shape[1] == 3:
        faces = topology.simplices(2)
        mesh = (
            {
                "type": "mesh3d",
                **_coordinates(points),
                "i": faces[:, 0],
                "j": faces[:, 1],
                "k": faces[:, 2],
                "opacity": 0.72,
                "name": "geometry",
            },
        )
    else:
        mesh = ()
    if topology.dimension >= 1:
        edges = points[topology.simplices(1)]
        separated = np.full((len(edges), 3, points.shape[1]), np.nan)
        separated[:, :2] = edges
        edge_traces = (
            _scatter(
                separated.reshape(-1, points.shape[1]),
                mode="lines",
                name="edges",
            ),
        )
    else:
        edge_traces = ()
    return (
        *mesh,
        *edge_traces,
        _scatter(points, mode="markers", name="vertices"),
    )


def _simplex_locations(
    topology: Complex, points: FloatArray, degree: int
) -> FloatArray:
    if degree == 0:
        return points
    return points[topology.simplices(degree)].mean(axis=1)


def _coefficient_trace(
    locations: FloatArray,
    coefficients: FloatArray,
    name: str,
) -> dict[str, object]:
    magnitude = float(np.max(np.abs(coefficients), initial=0.0)) or 1.0
    indices = np.arange(len(coefficients), dtype=np.int64)
    return _scatter(
        locations,
        mode="markers",
        name=name,
        customdata=np.column_stack((indices, coefficients)),
        marker={
            "color": coefficients,
            "colorscale": "RdBu",
            "cmin": -magnitude,
            "cmax": magnitude,
            "showscale": True,
        },
        hovertemplate=(
            "simplex %{customdata[0]}<br>coefficient %{customdata[1]}<extra></extra>"
        ),
    )


def _figure(traces: tuple[dict[str, object], ...], title: str | None) -> Figure:
    try:
        from plotly.graph_objects import Figure
    except ImportError as error:
        raise PlotError(
            "plotting requires the optional polygeo[plot] dependency"
        ) from error
    return Figure(
        data=traces,
        layout={"title": title, "template": "plotly_white"},
    )


def plot_geometry(
    geometry: EuclideanRealization,
    *,
    axes: PlotAxes | None = None,
    title: str | None = None,
) -> Figure:
    """Render one caller-owned realization snapshot."""
    topology, points = _snapshot(geometry, axes)
    return _figure(_geometry_traces(topology, points), title)


def plot_form(
    geometry: EuclideanRealization,
    form: Binary64Element,
    *,
    axes: PlotAxes | None = None,
    title: str | None = None,
) -> Figure:
    """Render a full native binary64 cochain over one geometry snapshot."""
    space = form.space
    if space.variance != "cochain" or not space.is_full:
        raise PlotError("form plot requires a full cochain space")
    topology, points = _snapshot(geometry, axes)
    if space.degree > topology.dimension or not space.same_space(
        topology.binary64_cochain_space(space.degree)
    ):
        raise PlotError("form and geometry belong to different cochain spaces")
    traces = (
        *_geometry_traces(topology, points),
        _coefficient_trace(
            _simplex_locations(topology, points, space.degree),
            form.coefficients_numpy_copy(),
            "form",
        ),
    )
    return _figure(traces, title)


def plot_homology_cycle(
    geometry: EuclideanRealization,
    group: HomologyGroup,
    cycle: int,
    *,
    axes: PlotAxes | None = None,
    title: str | None = None,
) -> Figure:
    """Render one selected free integral-homology representative."""
    if type(cycle) is not int or cycle < 0 or cycle >= group.free_rank:
        raise PlotError("homology plot requires an available free-cycle index")
    topology, points = _snapshot(geometry, axes)
    if group.degree > topology.dimension:
        raise PlotError("homology group and geometry have incompatible dimensions")
    try:
        realized = topology.binary64_chain_space(group.degree).realize_integral(
            group.free_cycle(cycle)
        )
    except ValueError as error:
        raise PlotError(
            "homology group and geometry belong to different complexes"
        ) from error
    traces = (
        *_geometry_traces(topology, points),
        _coefficient_trace(
            _simplex_locations(topology, points, group.degree),
            realized.coefficients_numpy_copy(),
            "homology cycle",
        ),
    )
    return _figure(traces, title)


def plot_surface_vectors(
    field: EntityVectors,
    *,
    scale: float = 1.0,
    axes: PlotAxes | None = None,
    title: str | None = None,
) -> Figure:
    """Render one geometry snapshot and one vector snapshot."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise PlotError(
            "surface vector plot requires a field and positive finite scale"
        )
    geometry = field.realization
    topology, points = _snapshot(geometry, axes)
    vectors = _project(field.vectors_numpy_copy(), axes)
    anchors = (
        points
        if field.is_vertex_supported
        else points[topology.simplices(2)].mean(axis=1)
    )
    segments = np.stack((anchors, anchors + scale * vectors), axis=1)
    separated = np.full((len(segments), 3, anchors.shape[1]), np.nan)
    separated[:, :2] = segments
    vector_trace = _scatter(
        separated.reshape(-1, anchors.shape[1]),
        mode="lines",
        name="vectors",
    )
    return _figure((*_geometry_traces(topology, points), vector_trace), title)


__all__ = [
    "PlotError",
    "plot_form",
    "plot_geometry",
    "plot_homology_cycle",
    "plot_surface_vectors",
]
