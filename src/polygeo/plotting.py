"""Optional Plotly adapters for complete PolyGeo mathematical values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.typing import NDArray

from .algorithms import RealHomologyBasis
from .geometry import Geometry, _GeometryDomain
from .operators import _OperatorDomain
from .simplicial import (
    CochainSpace,
    CochainSubspace,
    Complex,
    FieldSemantics,
    Form,
)
from .surface import FaceVectors, VertexVectors

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


type PlotAxes = tuple[int, int] | tuple[int, int, int]
type FloatArray = NDArray[np.float64]


class _PlotHomologyDomain(_GeometryDomain, _OperatorDomain, Protocol):
    pass


class PlotError(ValueError):
    """Invalid plotting relationship or unavailable optional plotting behavior."""


def plot_geometry[K: _GeometryDomain](
    geometry: Geometry[K],
    *,
    axes: PlotAxes | None = None,
    title: str | None = None,
) -> Figure:
    """Render canonical vertices and simplices without retaining plotting state."""
    try:
        from plotly import graph_objects as go
    except ImportError as error:
        raise PlotError(
            "plotting requires the optional polygeo[plot] dependency"
        ) from error

    if not isinstance(geometry, Geometry):
        raise PlotError("geometry plot requires a Geometry")
    _require_title(title)
    coordinates = _display_coordinates(geometry, axes)
    complex_ = geometry.complex
    if complex_.dimension > 2:
        raise PlotError("plotting supports intrinsic dimension at most two")

    traces: list = []
    display_dimension = coordinates.shape[1]
    if complex_.dimension == 2 and display_dimension == 3:
        faces = complex_.simplices(2)
        traces.append(
            go.Mesh3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#bdd7e7",
                opacity=0.72,
                flatshading=True,
                hoverinfo="skip",
                name="geometry",
                showscale=False,
            )
        )
    if complex_.dimension >= 1:
        edges = complex_.simplices(1)
        edge_coordinates = coordinates[edges]
        flattened = _separated_segments(edge_coordinates)
        line = {"color": "#52616b", "width": 2}
        if display_dimension == 2:
            traces.append(
                go.Scatter(
                    x=flattened[:, 0],
                    y=flattened[:, 1],
                    mode="lines",
                    line=line,
                    hoverinfo="skip",
                    name="edges",
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter3d(
                    x=flattened[:, 0],
                    y=flattened[:, 1],
                    z=flattened[:, 2],
                    mode="lines",
                    line=line,
                    hoverinfo="skip",
                    name="edges",
                    showlegend=False,
                )
            )
        edge_indices = np.arange(complex_.simplex_count(1), dtype=np.int64)
        edge_locations = edge_coordinates.mean(axis=1)
        marker = {"color": "rgba(0,0,0,0)", "size": 12}
        hover = "edge %{customdata}<extra></extra>"
        if display_dimension == 2:
            traces.append(
                go.Scatter(
                    x=edge_locations[:, 0],
                    y=edge_locations[:, 1],
                    mode="markers",
                    marker=marker,
                    customdata=edge_indices,
                    hovertemplate=hover,
                    name="edge indices",
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter3d(
                    x=edge_locations[:, 0],
                    y=edge_locations[:, 1],
                    z=edge_locations[:, 2],
                    mode="markers",
                    marker=marker,
                    customdata=edge_indices,
                    hovertemplate=hover,
                    name="edge indices",
                    showlegend=False,
                )
            )

    if complex_.dimension == 2:
        faces = complex_.simplices(2)
        face_indices = np.arange(complex_.simplex_count(2), dtype=np.int64)
        face_locations = coordinates[faces].mean(axis=1)
        marker = {"color": "rgba(0,0,0,0)", "size": 14}
        hover = "face %{customdata}<extra></extra>"
        if display_dimension == 2:
            traces.append(
                go.Scatter(
                    x=face_locations[:, 0],
                    y=face_locations[:, 1],
                    mode="markers",
                    marker=marker,
                    customdata=face_indices,
                    hovertemplate=hover,
                    name="face indices",
                    showlegend=False,
                )
            )
        else:
            traces.append(
                go.Scatter3d(
                    x=face_locations[:, 0],
                    y=face_locations[:, 1],
                    z=face_locations[:, 2],
                    mode="markers",
                    marker=marker,
                    customdata=face_indices,
                    hovertemplate=hover,
                    name="face indices",
                    showlegend=False,
                )
            )

    indices = np.arange(complex_.vertex_count, dtype=np.int64)
    marker = {"color": "#243b53", "size": 7}
    hover = "vertex %{customdata}<extra></extra>"
    if display_dimension == 2:
        traces.append(
            go.Scatter(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                mode="markers",
                marker=marker,
                customdata=indices,
                hovertemplate=hover,
                name="vertices",
                showlegend=False,
            )
        )
        figure = go.Figure(data=traces)
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        traces.append(
            go.Scatter3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                mode="markers",
                marker=marker,
                customdata=indices,
                hovertemplate=hover,
                name="vertices",
                showlegend=False,
            )
        )
        figure = go.Figure(data=traces)
        figure.update_scenes(aspectmode="data")
    figure.update_layout(
        autosize=True,
        title=title,
        template="plotly_white",
        margin={"l": 24, "r": 24, "t": 56 if title else 24, "b": 24},
    )
    return figure


def plot_surface_vectors[K: Complex](
    field: VertexVectors[K] | FaceVectors[K],
    *,
    scale: float = 1.0,
    axes: PlotAxes | None = None,
    title: str | None = None,
) -> Figure:
    """Render exact-geometry-bound ambient vectors at vertices or triangle faces."""
    try:
        from plotly import graph_objects as go
    except ImportError as error:
        raise PlotError(
            "plotting requires the optional polygeo[plot] dependency"
        ) from error

    if not isinstance(field, (VertexVectors, FaceVectors)):
        raise PlotError("surface vector plot requires surface vectors")
    if type(scale) is not float or not np.isfinite(scale) or scale <= 0.0:
        raise PlotError("surface vector scale must be finite and positive")
    _require_title(title)

    geometry = field.geometry
    _display_coordinates(geometry, axes)
    if isinstance(field, VertexVectors):
        anchors = geometry.positions
    else:
        anchors = geometry.positions[geometry.complex.simplices(2)].mean(axis=1)
    vectors = field.vectors
    if axes is None:
        projected_anchors = anchors
        projected_vectors = vectors
    else:
        projected_anchors = anchors[:, axes]
        projected_vectors = vectors[:, axes]
    if projected_anchors.shape[1] not in (2, 3):
        raise PlotError("surface vector plotting requires two or three display axes")

    directions = scale * projected_vectors
    segments = np.stack((projected_anchors, projected_anchors + directions), axis=1)
    flattened = _separated_segments(segments)
    figure = plot_geometry(geometry, axes=axes, title=title)
    line = {"color": "#b2182b", "width": 4}
    if flattened.shape[1] == 2:
        trace = go.Scatter(
            x=flattened[:, 0],
            y=flattened[:, 1],
            mode="lines",
            line=line,
            hoverinfo="skip",
            name="surface vectors",
            showlegend=False,
        )
    else:
        trace = go.Scatter3d(
            x=flattened[:, 0],
            y=flattened[:, 1],
            z=flattened[:, 2],
            mode="lines",
            line=line,
            hoverinfo="skip",
            name="surface vectors",
            showlegend=False,
        )
    figure.add_trace(trace)
    if flattened.shape[1] == 3:
        nonzero = np.any(directions != 0.0, axis=1)
        tips = segments[nonzero, 1]
        head_directions = directions[nonzero]
        if len(tips):
            figure.add_trace(
                go.Cone(
                    x=tips[:, 0],
                    y=tips[:, 1],
                    z=tips[:, 2],
                    u=head_directions[:, 0],
                    v=head_directions[:, 1],
                    w=head_directions[:, 2],
                    anchor="tip",
                    sizemode="absolute",
                    sizeref=0.36 * scale,
                    colorscale=[[0.0, "#b2182b"], [1.0, "#b2182b"]],
                    showscale=False,
                    hoverinfo="skip",
                    name="surface vector arrowheads",
                    showlegend=False,
                )
            )
    return figure


def plot_cochain[
    K: _GeometryDomain,
    Degree: int,
    Semantics: FieldSemantics,
](
    geometry: Geometry[K],
    form: Form[CochainSpace[K, Degree], Semantics]
    | Form[CochainSubspace[CochainSpace[K, Degree]], Semantics],
    *,
    axes: PlotAxes | None = None,
    title: str | None = None,
) -> Figure:
    """Render cochain coefficients on their exact canonical simplices."""
    try:
        from plotly import graph_objects as go
    except ImportError as error:
        raise PlotError(
            "plotting requires the optional polygeo[plot] dependency"
        ) from error

    if not isinstance(geometry, Geometry) or not isinstance(form, Form):
        raise PlotError("cochain plot requires a Geometry and Form")
    _require_title(title)
    space = form.space
    if isinstance(space, CochainSubspace):
        parent = space.parent
        selected = space.indices()
    elif isinstance(space, CochainSpace):
        parent = space
        selected = np.arange(space.size, dtype=np.int64)
    else:
        raise PlotError("cochain plot requires a primal cochain space")
    if parent.complex is not geometry.complex:
        raise PlotError("cochain and geometry belong to a different complex")

    degree = parent.degree
    if degree < 0 or degree > min(2, geometry.complex.dimension):
        raise PlotError("cochain degree is outside the supported plotting dimension")
    coefficients = form.coefficients()
    coordinates = _display_coordinates(geometry, axes)
    figure = plot_geometry(geometry, axes=axes, title=title)
    locations = _simplex_locations(geometry, coordinates, degree, selected)
    marker = _coefficient_marker(coefficients)
    customdata = selected
    hover = f"degree-{degree} simplex %{{customdata}}<br>coefficient %{{marker.color}}<extra></extra>"
    if degree == 2:
        faces = geometry.complex.simplices(2)[selected]
        maximum = float(np.max(np.abs(coefficients), initial=0.0))
        if maximum == 0.0:
            maximum = 1.0
        if coordinates.shape[1] == 3:
            figure.add_trace(
                go.Mesh3d(
                    x=coordinates[:, 0],
                    y=coordinates[:, 1],
                    z=coordinates[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    intensity=coefficients,
                    intensitymode="cell",
                    colorscale="RdBu",
                    cmin=-maximum,
                    cmax=maximum,
                    cmid=0.0,
                    customdata=np.column_stack([selected, coefficients]),
                    hoverinfo="skip",
                    name="coefficient faces",
                    showscale=False,
                    showlegend=False,
                    opacity=0.82,
                )
            )
        else:
            for index, (face, coefficient) in enumerate(
                zip(faces, coefficients, strict=True)
            ):
                points = coordinates[np.append(face, face[0])]
                color = (
                    "rgba(178,24,43,0.55)"
                    if coefficient < 0
                    else "rgba(33,102,172,0.55)"
                )
                figure.add_trace(
                    go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode="lines",
                        fill="toself",
                        fillcolor=color,
                        line={"color": color, "width": 1},
                        customdata=np.repeat(
                            [[selected[index], coefficient]], len(points), axis=0
                        ),
                        hovertemplate="face %{customdata[0]}<br>coefficient %{customdata[1]}<extra></extra>",
                        name="coefficient faces" if index == 0 else None,
                        showlegend=False,
                    )
                )
    if coordinates.shape[1] == 2:
        trace = go.Scatter(
            x=locations[:, 0],
            y=locations[:, 1],
            mode="markers",
            marker=marker,
            customdata=customdata,
            hovertemplate=hover,
            name="coefficients",
            showlegend=False,
        )
    else:
        trace = go.Scatter3d(
            x=locations[:, 0],
            y=locations[:, 1],
            z=locations[:, 2],
            mode="markers",
            marker=marker,
            customdata=customdata,
            hovertemplate=hover,
            name="coefficients",
            showlegend=False,
        )
    figure.add_trace(trace)
    return figure


def plot_homology_cycle[K: _PlotHomologyDomain, Degree: int](
    geometry: Geometry[K],
    basis: RealHomologyBasis[K, Degree],
    cycle: int,
    *,
    axes: PlotAxes | None = None,
    title: str | None = None,
) -> Figure:
    """Render one chain-owned sparse homology representative."""
    try:
        from plotly import graph_objects as go
    except ImportError as error:
        raise PlotError(
            "plotting requires the optional polygeo[plot] dependency"
        ) from error

    if not isinstance(geometry, Geometry) or not isinstance(basis, RealHomologyBasis):
        raise PlotError("cycle plot requires a Geometry and RealHomologyBasis")
    _require_title(title)
    if basis.complex is not geometry.complex:
        raise PlotError("homology basis and geometry belong to a different complex")
    if type(cycle) is not int or cycle < 0 or cycle >= basis.dimension:
        raise PlotError("cycle index is outside the homology basis")
    degree = basis.degree
    if degree < 0 or degree > min(2, geometry.complex.dimension):
        raise PlotError("cycle degree is outside the supported plotting dimension")

    coefficients = basis.cycle_coefficients()[:, cycle].toarray().reshape(-1)
    selected = np.flatnonzero(coefficients)
    coordinates = _display_coordinates(geometry, axes)
    figure = plot_geometry(geometry, axes=axes, title=title)
    locations = _simplex_locations(geometry, coordinates, degree, selected)
    values = coefficients[selected]
    customdata = np.column_stack([selected, values])
    marker = _coefficient_marker(values)
    hover = (
        "simplex %{customdata[0]}<br>chain coefficient %{customdata[1]}<extra></extra>"
    )

    if degree == 1:
        edges = geometry.complex.simplices(1)[selected]
        for edge, coefficient in zip(edges, values, strict=True):
            points = coordinates[edge]
            color = "#b2182b" if coefficient < 0 else "#2166ac"
            if coordinates.shape[1] == 2:
                figure.add_trace(
                    go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode="lines",
                        line={"color": color, "width": 6},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            else:
                figure.add_trace(
                    go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        mode="lines",
                        line={"color": color, "width": 6},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
    if degree == 2:
        faces = geometry.complex.simplices(2)[selected]
        colors = np.where(values < 0, "#b2182b", "#2166ac")
        if coordinates.shape[1] == 3:
            figure.add_trace(
                go.Mesh3d(
                    x=coordinates[:, 0],
                    y=coordinates[:, 1],
                    z=coordinates[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    facecolor=colors,
                    customdata=customdata,
                    hoverinfo="skip",
                    name="cycle faces",
                    showlegend=False,
                    opacity=0.82,
                )
            )
        else:
            for index, (face, color) in enumerate(zip(faces, colors, strict=True)):
                points = coordinates[np.append(face, face[0])]
                figure.add_trace(
                    go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode="lines",
                        fill="toself",
                        fillcolor=color,
                        line={"color": color, "width": 2},
                        customdata=np.repeat(
                            customdata[index : index + 1], len(points), axis=0
                        ),
                        hovertemplate=hover,
                        name="cycle faces" if index == 0 else None,
                        showlegend=False,
                    )
                )
    if coordinates.shape[1] == 2:
        trace = go.Scatter(
            x=locations[:, 0],
            y=locations[:, 1],
            mode="markers",
            marker=marker,
            customdata=customdata,
            hovertemplate=hover,
            name="cycle",
            showlegend=False,
        )
    else:
        trace = go.Scatter3d(
            x=locations[:, 0],
            y=locations[:, 1],
            z=locations[:, 2],
            mode="markers",
            marker=marker,
            customdata=customdata,
            hovertemplate=hover,
            name="cycle",
            showlegend=False,
        )
    figure.add_trace(trace)
    return figure


def _display_coordinates(
    geometry: Geometry[_GeometryDomain], axes: PlotAxes | None
) -> FloatArray:
    positions = geometry.positions
    ambient = positions.shape[1]
    if axes is None:
        if ambient == 0:
            return np.zeros((len(positions), 2), dtype=np.float64)
        if ambient == 1:
            return np.column_stack([positions[:, 0], np.zeros(len(positions))])
        if ambient in (2, 3):
            return positions
        raise PlotError(
            "an explicit projection is required above ambient dimension three"
        )
    if (
        type(axes) is not tuple
        or len(axes) not in (2, 3)
        or any(type(axis) is not int for axis in axes)
        or len(set(axes)) != len(axes)
        or any(axis < 0 or axis >= ambient for axis in axes)
    ):
        raise PlotError("projection axes must be distinct in-range built-in integers")
    return positions[:, axes]


def _separated_segments(segments: FloatArray) -> FloatArray:
    separated = np.full(
        (len(segments), segments.shape[1] + 1, segments.shape[2]), np.nan
    )
    separated[:, : segments.shape[1]] = segments
    return separated.reshape(-1, segments.shape[2])


def _simplex_locations(
    geometry: Geometry[_GeometryDomain],
    coordinates: FloatArray,
    degree: int,
    selected: NDArray[np.int64],
) -> FloatArray:
    if degree == 0:
        return coordinates[selected]
    simplices = geometry.complex.simplices(degree)[selected]
    return coordinates[simplices].mean(axis=1)


def _coefficient_marker(values: FloatArray) -> dict[str, object]:
    maximum = float(np.max(np.abs(values), initial=0.0))
    if maximum == 0.0:
        maximum = 1.0
    return {
        "color": values,
        "colorscale": "RdBu",
        "cmin": -maximum,
        "cmax": maximum,
        "cmid": 0.0,
        "colorbar": {"title": "coefficient"},
        "size": 10,
        "line": {"color": "#102a43", "width": 0.5},
    }


def _require_title(title: str | None) -> None:
    if title is not None and type(title) is not str:
        raise PlotError("plot title must be a string or None")


__all__ = [
    "PlotError",
    "plot_cochain",
    "plot_geometry",
    "plot_homology_cycle",
    "plot_surface_vectors",
]
