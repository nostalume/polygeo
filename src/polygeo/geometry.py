"""Admitted Euclidean geometry and surface operators."""

from ._polygeo_native import geometry as _native

Geometry = _native.Geometry
Metric = _native.Metric
Limit = _native.Limit
DEFAULT_LIMIT = _native.DEFAULT_LIMIT
GeometryError = _native.GeometryError
SurfaceError = _native.SurfaceError
TriangleSurface = _native.TriangleSurface
VectorField = _native.VectorField
VertexField = _native.VertexField
FaceField = _native.FaceField
FlowStep = _native.FlowStep
ConformalMap = _native.ConformalMap

__all__ = [
    "Geometry",
    "Metric",
    "Limit",
    "DEFAULT_LIMIT",
    "GeometryError",
    "SurfaceError",
    "TriangleSurface",
    "VectorField",
    "VertexField",
    "FaceField",
    "FlowStep",
    "ConformalMap",
]
