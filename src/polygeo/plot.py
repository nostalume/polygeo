"""Optional Plotly output effects over explicit native snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .chain import HomologyGroup
from .field import Direction
from .form import Element
from .geometry import Geometry, VectorField
from .topology import Complex

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

type PlotAxes = tuple[int, int] | tuple[int, int, int]
type FloatArray = NDArray[np.float64]

_COEFFICIENT_FIGURE_LAYOUT: dict[str, object] = {
    "legend": {
        "orientation": "h",
        "x": 0.5,
        "xanchor": "center",
        "y": -0.15,
        "yanchor": "top",
    },
    "margin": {"b": 100},
}


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


def _snapshot(geometry: Geometry, axes: PlotAxes | None) -> tuple[Complex, FloatArray]:
    topology = geometry.topology
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
        faces = topology.simplices_numpy_copy(2)
        mesh = (
            {
                "type": "mesh3d",
                **_coordinates(points),
                "i": faces[:, 0],
                "j": faces[:, 1],
                "k": faces[:, 2],
                "color": "#d1d5db",
                "opacity": 0.35,
                "name": "geometry",
            },
        )
    else:
        mesh = ()
    if topology.dimension >= 1:
        edges = points[topology.simplices_numpy_copy(1)]
        separated = np.full((len(edges), 3, points.shape[1]), np.nan)
        separated[:, :2] = edges
        edge_traces = (
            _scatter(
                separated.reshape(-1, points.shape[1]),
                mode="lines",
                name="edges",
                line={"color": "#4b5563", "width": 1.5},
            ),
        )
    else:
        edge_traces = ()
    return (
        *mesh,
        *edge_traces,
        _scatter(
            points,
            mode="markers",
            name="vertices",
            marker={"color": "#6b7280", "size": 3},
        ),
    )


def _simplex_locations(
    topology: Complex, points: FloatArray, degree: int
) -> FloatArray:
    if degree == 0:
        return points
    return points[topology.simplices_numpy_copy(degree)].mean(axis=1)


def _coefficient_trace(
    locations: FloatArray,
    coefficients: FloatArray,
    degree: int,
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
            "size": (5, 5, 6)[degree],
            "symbol": ("circle", "diamond", "square")[degree],
            "colorbar": {
                "orientation": "v",
                "x": 1.02,
                "xanchor": "left",
                "title": {"text": name},
            },
        },
        hovertemplate=(
            "simplex %{customdata[0]}<br>coefficient %{customdata[1]}<extra></extra>"
        ),
    )


def _segments(anchors: FloatArray, vectors: FloatArray, scale: float) -> FloatArray:
    segments = np.stack((anchors, anchors + scale * vectors), axis=1)
    separated = np.full((len(segments), 3, anchors.shape[1]), np.nan)
    separated[:, :2] = segments
    return separated.reshape(-1, anchors.shape[1])


def _figure(
    traces: tuple[dict[str, object], ...],
    title: str | None,
    *,
    layout: dict[str, object] | None = None,
) -> Figure:
    try:
        from plotly.graph_objects import Figure
    except ImportError as error:
        raise PlotError(
            "plotting requires the optional polygeo[plot] dependency"
        ) from error
    return Figure(
        data=traces,
        layout={
            "title": title,
            "template": "plotly_white",
            **({} if layout is None else layout),
        },
    )


def geometry(
    geometry: Geometry,
    *,
    axes: PlotAxes | None = None,
    title: str | None = None,
) -> Figure:
    """Render one caller-owned realization snapshot."""
    topology, points = _snapshot(geometry, axes)
    return _figure(_geometry_traces(topology, points), title)


def form(
    geometry: Geometry,
    form: Element,
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
            space.degree,
            "form",
        ),
    )
    return _figure(traces, title, layout=_COEFFICIENT_FIGURE_LAYOUT)


def homology_cycle(
    geometry: Geometry,
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
            group.degree,
            "homology cycle",
        ),
    )
    return _figure(traces, title, layout=_COEFFICIENT_FIGURE_LAYOUT)


def vectors(
    field: VectorField[int],
    *,
    scale: float = 1.0,
    axes: PlotAxes | None = None,
    title: str | None = None,
) -> Figure:
    """Render one geometry snapshot and one vector snapshot."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise PlotError("surface vectors require a positive finite scale")
    geometry = field.geometry
    topology, points = _snapshot(geometry, axes)
    vectors = _project(field.values_numpy_copy(), axes)
    anchors = _simplex_locations(topology, points, field.support_degree)
    vector_trace = _scatter(
        _segments(anchors, vectors, scale),
        mode="lines",
        name="vectors",
        legendgroup="vectors",
        line={"color": "#d97706", "width": 2},
        showlegend=True,
    )
    tips = anchors + scale * vectors
    lengths = np.linalg.norm(vectors, axis=1)
    nonzero = lengths > 0.0
    if anchors.shape[1] == 2:
        angles = 90.0 - np.degrees(np.arctan2(vectors[nonzero, 1], vectors[nonzero, 0]))
        head_trace = _scatter(
            tips[nonzero],
            mode="markers",
            name="vector heads",
            legendgroup="vectors",
            marker={
                "angle": angles,
                "angleref": "up",
                "color": "#d97706",
                "size": 8,
                "symbol": "triangle-up",
            },
            hoverinfo="skip",
            showlegend=False,
        )
    else:
        directions = vectors[nonzero] / lengths[nonzero, None]
        head_trace = {
            "type": "cone",
            **_coordinates(tips[nonzero]),
            "u": directions[:, 0],
            "v": directions[:, 1],
            "w": directions[:, 2],
            "anchor": "tip",
            "colorscale": [[0.0, "#d97706"], [1.0, "#d97706"]],
            "hoverinfo": "skip",
            "legendgroup": "vectors",
            "name": "vector heads",
            "showlegend": False,
            "showscale": False,
            "sizemode": "absolute",
            "sizeref": 0.18 * scale,
        }
    return _figure(
        (*_geometry_traces(topology, points), vector_trace, head_trace), title
    )


def direction(
    field: Direction,
    *,
    scale: float = 1.0,
    axes: PlotAxes | None = None,
    title: str | None = None,
) -> Figure:
    """Render every equivalent branch of one symmetric face direction field."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise PlotError("direction-field plot requires a positive finite scale")
    topology, points = _snapshot(field.connection.connection.surface.geometry, axes)
    anchors = _simplex_locations(topology, points, 2)
    order = field.symmetry_order
    branches = tuple(
        _scatter(
            _segments(
                anchors,
                _project(
                    field.ambient_branch_numpy_copy(branch).values_numpy_copy(),
                    axes,
                ),
                scale,
            ),
            mode="lines",
            name=f"branch {branch + 1}/{order}",
            line={"color": "#2563eb", "width": 1.5},
            hoverinfo="skip",
        )
        for branch in range(order)
    )
    return _figure((*_geometry_traces(topology, points), *branches), title)


__all__ = [
    "PlotError",
    "geometry",
    "form",
    "vectors",
    "direction",
    "homology_cycle",
]
