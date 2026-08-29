"""Stable public identities, names, and retired-import boundaries."""

from __future__ import annotations

import importlib.util

import polygeo
from polygeo import _polygeo_native as native


def test_public_mathematical_objects_have_one_package_identity() -> None:
    for public_name, native_name in (
        ("Complex", "Complex"),
        ("SimplexSubset", "SimplexSubset"),
        ("SimplexSelection", "SimplexSelection"),
        ("SimplicialError", "SimplicialError"),
        ("HalfedgeSurface", "HalfedgeSurface"),
        ("SurfaceCorrespondence", "SurfaceCorrespondence"),
        ("HalfedgeError", "HalfedgeError"),
        ("ChainComplex", "ChainComplex"),
        ("IntegralChainComplex", "ChainComplex"),
        ("RationalChainComplex", "ChainComplex"),
        ("CsrRepresentation", "CsrRepresentation"),
        ("HomologyLimit", "HomologyLimit"),
        ("IntegralHomology", "IntegralHomology"),
        ("HomologyGroup", "HomologyGroup"),
        ("prepare_integral_homology", "prepare_integral_homology"),
    ):
        assert getattr(polygeo, public_name) is getattr(native, native_name)

    for public_object in (
        polygeo.Complex,
        polygeo.HalfedgeSurface,
        polygeo.ChainError,
        polygeo.ChainComplex,
        polygeo.CsrRepresentation,
        polygeo.HomologyError,
        polygeo.IntegralHomology,
        polygeo.prepare_integral_homology,
    ):
        assert public_object.__module__ == "polygeo"


def test_allocating_boundaries_name_admission_and_owned_copies() -> None:
    assert hasattr(polygeo.Complex, "from_maximal_simplices")
    assert not hasattr(polygeo.Complex, "admit_numpy")

    assert hasattr(polygeo.IntegralChain, "to_python_copy")
    assert not hasattr(polygeo.IntegralChain, "to_python_parts")
    assert hasattr(polygeo.CsrRepresentation, "to_python_copy")
    assert hasattr(polygeo.CsrRepresentation, "to_scipy_int64_copy")
    assert not hasattr(polygeo.CsrRepresentation, "to_python_parts")
    assert not hasattr(polygeo.CsrRepresentation, "to_scipy_int64")

    for private_name in (
        "NativeHalfedgeSurface",
        "NativeSurfaceCorrespondence",
        "NativeChainComplex",
        "NativeChainSpace",
        "NativeChainElement",
        "NativeLinearMap",
        "NativeCsrRepresentation",
        "NativeHomologyError",
        "NativeIntegralHomology",
        "NativeHomologyGroup",
    ):
        assert not hasattr(native, private_name)


def test_retired_qualified_modules_are_not_importable() -> None:
    for qualified_name in (
        "polygeo._native",
        "polygeo._topology_runtime",
        "polygeo.algorithms",
        "polygeo.chain",
        "polygeo.geometry",
        "polygeo.halfedge",
        "polygeo.homology",
        "polygeo.numerics",
        "polygeo.operators",
        "polygeo.simplicial",
        "polygeo.solvers",
        "polygeo.surface",
        "polygeo.systems",
    ):
        assert importlib.util.find_spec(qualified_name) is None
