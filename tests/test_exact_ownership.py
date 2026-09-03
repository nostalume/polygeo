"""Exact values retain owners and publish only caller-owned copies."""

from __future__ import annotations

import gc
import numpy as np

from polygeo.chain import (
    BigIntEncoding,
    ChainComplex,
    ChainIsomorphism,
    CsrBuildLimit,
    CsrEstimate,
    Csr,
    Element,
    IntegerCsrParts,
    LinearMap,
    Space,
)
from polygeo.topology import Complex


def test_exact_carriers_are_sealed_and_not_partially_constructible() -> None:
    for carrier in (
        ChainComplex,
        ChainIsomorphism,
        Space,
        Element,
        LinearMap,
        CsrEstimate,
        CsrBuildLimit,
        Csr,
        IntegerCsrParts,
    ):
        with np.testing.assert_raises(TypeError):
            carrier()
        with np.testing.assert_raises(TypeError):
            type("Derived", (carrier,), {})


def test_derived_handles_survive_every_originating_owner() -> None:
    domain = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
    complex_ = domain.chain_complex()
    space = complex_[1]
    value = space.element({0: 4})
    map_ = complex_.boundary(1)
    estimate = Csr.estimate(map_, BigIntEncoding)
    representation = Csr.build(map_, BigIntEncoding, estimate.as_limit())
    assert not hasattr(complex_, "space")
    assert not hasattr(map_, "estimate_csr")
    assert not hasattr(map_, "build_csr")
    assert not hasattr(representation, "to_scipy_int64_parts")

    del domain, complex_, space, map_
    gc.collect()

    assert value.to_python_copy() == ((0,), (4,))
    assert representation.apply(value).to_python_copy() == ((0, 1), (-4, 4))


def test_exact_handles_are_frozen_and_projections_do_not_alias() -> None:
    complex_ = Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64)
    ).chain_complex()
    space = complex_[1]
    value = space.element({0: 7})
    boundary = complex_.boundary(1)
    estimate = Csr.estimate(boundary, BigIntEncoding)
    representation = Csr.build(boundary, BigIntEncoding, estimate.as_limit())

    for handle in (complex_, space, value, representation):
        try:
            setattr(handle, "changed", True)
        except AttributeError:
            pass
        else:
            raise AssertionError("native exact handles must be frozen")

    indices, coefficients = value.to_python_copy()
    indices += (99,)
    coefficients += (99,)
    assert value.to_python_copy() == ((0,), (7,))

    with np.testing.assert_raises(AttributeError):
        setattr(complex_, "changed", True)
