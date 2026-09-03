"""Optional Trimesh input effect for native PolyGeo owners."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import Geometry
from .topology import Complex


class MeshError(ValueError):
    """Invalid mesh input or unavailable optional mesh-loading behavior."""


def load_surface(source: str | PathLike[str]) -> Geometry:
    """Load one triangular mesh into owned native topology and geometry."""
    try:
        import trimesh
    except (ModuleNotFoundError, ImportError) as error:
        raise MeshError(
            "mesh input requires the optional polygeo[mesh] dependency"
        ) from error

    try:
        payload: Any = trimesh.load(Path(source), process=False)
        if not isinstance(payload, trimesh.Trimesh):
            raise TypeError
        positions = np.asarray(payload.vertices)
        faces = np.asarray(payload.faces)
        complex_ = Complex.from_maximal_simplices(faces, vertex_count=len(positions))
        return Geometry.from_positions(complex_, positions)
    except Exception as error:
        raise MeshError("surface mesh is not admissible") from error


__all__ = ["MeshError", "load_surface"]
