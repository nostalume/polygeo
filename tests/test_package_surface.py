"""Canonical contextual package boundary and public identities."""

from __future__ import annotations

import polygeo
from polygeo import chain, field, form, geometry, mesh, plot, solve, topology


def test_package_root_exports_only_the_canonical_modules() -> None:
    assert polygeo.__all__ == [
        "topology",
        "chain",
        "form",
        "geometry",
        "solve",
        "field",
        "plot",
        "mesh",
    ]


def test_public_objects_have_one_contextual_identity() -> None:
    for owner, public_name in (
        (topology, "Complex"),
        (topology, "Subset"),
        (topology, "Selection"),
        (topology, "HalfedgeSurface"),
        (chain, "ChainComplex"),
        (chain, "Csr"),
        (form, "Space"),
        (form, "Element"),
        (form, "Operator"),
        (geometry, "Geometry"),
        (geometry, "TriangleSurface"),
        (solve, "Policy"),
        (solve, "Prepared"),
        (field, "HarmonicBasis"),
        (field, "Direction"),
    ):
        value = getattr(owner, public_name)
        assert value.__module__ == owner.__name__


def test_allocating_operations_name_their_owned_copies() -> None:
    assert hasattr(topology.Complex, "simplices_numpy_copy")
    assert hasattr(topology.Subset, "mask_numpy_copy")
    assert hasattr(topology.Selection, "indices_numpy_copy")
    assert hasattr(chain.Csr, "to_python_copy")
    assert hasattr(chain.Csr, "to_scipy_int64_copy")
    assert hasattr(form.Element, "coefficients_numpy_copy")
    assert hasattr(geometry.Geometry, "positions_numpy_copy")
    assert hasattr(geometry.VectorField, "values_numpy_copy")
    assert hasattr(field.Direction, "ambient_branch_numpy_copy")
    assert hasattr(field.Singularities, "boundary_turns_python_copy")


def test_effect_leaves_are_modules_not_root_aliases() -> None:
    assert mesh.__all__ == ["MeshError", "load_surface"]
    assert plot.__all__ == [
        "PlotError",
        "geometry",
        "form",
        "vectors",
        "direction",
        "homology_cycle",
    ]
