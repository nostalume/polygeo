"""Connection cases shared by the holonomy study and its fixture tests."""

from __future__ import annotations

from typing import Literal

import numpy as np

from polygeo import (
    Geometry,
    SurfaceConnection,
    levi_civita_connection,
    surface_connection,
)


type ConnectionCase = Literal["levi-civita", "cancelled", "quarter-turn"]


def torus_connection(geometry: Geometry, case: ConnectionCase) -> SurfaceConnection:
    """Construct one declared represented SO(2) connection on an analytic torus."""
    levi_civita = levi_civita_connection(geometry)
    if case == "levi-civita":
        return levi_civita
    products = levi_civita.transport_products()
    if case == "cancelled":
        deviations = -np.angle(products)
    elif case == "quarter-turn":
        dual_edges = levi_civita.dual_edges()
        faces = geometry.complex.simplices(2)
        centers = geometry.positions[faces].mean(axis=1)
        major_angles = np.arctan2(centers[:, 1], centers[:, 0])
        raw = major_angles[dual_edges[:, 1]] - major_angles[dual_edges[:, 0]]
        winding = np.arctan2(np.sin(raw), np.cos(raw))
        quarter_transport = np.exp(0.25j * winding)
        deviations = np.angle(quarter_transport * np.conjugate(products))
    else:
        raise ValueError(f"unknown connection case: {case}")
    return surface_connection(geometry, deviations.astype(np.float64))


__all__ = ["ConnectionCase", "torus_connection"]
