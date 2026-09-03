"""Admitted simplicial and halfedge topology."""

from ._polygeo_native import topology as _native

Complex = _native.Complex
Subset = _native.Subset
Selection = _native.Selection
HalfedgeSurface = _native.HalfedgeSurface
SimplicialError = _native.SimplicialError
HalfedgeError = _native.HalfedgeError
topological_boundary = _native.topological_boundary

__all__ = [
    "Complex",
    "Subset",
    "Selection",
    "HalfedgeSurface",
    "SimplicialError",
    "HalfedgeError",
    "topological_boundary",
]
