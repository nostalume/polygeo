"""Hodge, harmonic, connection, holonomy, and direction fields."""

from ._polygeo_native import field as _native

DualCycles = _native.DualCycles
HodgeDecomposition = _native.HodgeDecomposition
HarmonicBasis = _native.HarmonicBasis
Connection = _native.Connection
Holonomy = _native.Holonomy
IntegrableConnection = _native.IntegrableConnection
Direction = _native.Direction
Singularities = _native.Singularities

__all__ = [
    "DualCycles",
    "HodgeDecomposition",
    "HarmonicBasis",
    "Connection",
    "Holonomy",
    "IntegrableConnection",
    "Direction",
    "Singularities",
]
