"""Exact coefficients, chains, maps, homology, and bounded CSR projections."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any

from ._polygeo_native import chain as _native

IntegerRing = _native.IntegerRing
RationalField = _native.RationalField
ZZ = _native.ZZ
QQ = _native.QQ
Chain = _native.Chain
Cochain = _native.Cochain
BigIntEncoding = _native.BigIntEncoding
ReducedFractionEncoding = _native.ReducedFractionEncoding
Space = _native.Space
Element = _native.Element
LinearMap = _native.LinearMap
ChainComplex = _native.ChainComplex
CochainComplex = _native.CochainComplex
ChainIsomorphism = _native.ChainIsomorphism
ChainError = _native.ChainError
ChainLawLimit = _native.ChainLawLimit
DEFAULT_LAW_LIMIT = _native.DEFAULT_LAW_LIMIT
CsrEstimate = _native.CsrEstimate
CsrBuildLimit = _native.CsrBuildLimit
Csr = _native.Csr
IntegerCsrParts = _native.IntegerCsrParts
RationalCsrParts = _native.RationalCsrParts
HomologyError = _native.HomologyError
HomologyLimit = _native.HomologyLimit
DEFAULT_HOMOLOGY_LIMIT = _native.DEFAULT_HOMOLOGY_LIMIT
HomologyGroup = _native.HomologyGroup
IntegralHomology = _native.IntegralHomology
analyze_integral_homology = _native.analyze_integral_homology

if TYPE_CHECKING:
    type IntegralChainComplex = ChainComplex[int]
    type RationalChainComplex = ChainComplex[Fraction]
    type IntegralCochainComplex = CochainComplex[int]
    type RationalCochainComplex = CochainComplex[Fraction]
    type IntegralChainSpace[Degree: int] = Space[int, Chain, Degree]
    type RationalChainSpace[Degree: int] = Space[Fraction, Chain, Degree]
    type IntegralCochainSpace[Degree: int] = Space[int, Cochain, Degree]
    type RationalCochainSpace[Degree: int] = Space[Fraction, Cochain, Degree]
    type IntegralChain[Degree: int] = Element[IntegralChainSpace[Degree]]
    type RationalChain[Degree: int] = Element[RationalChainSpace[Degree]]
    type IntegralCochain[Degree: int] = Element[IntegralCochainSpace[Degree]]
    type RationalCochain[Degree: int] = Element[RationalCochainSpace[Degree]]
    type IntegralLinearMap[
        SourceSpace: Space[int, Any, int],
        TargetSpace: Space[int, Any, int],
    ] = LinearMap[SourceSpace, TargetSpace]
    type RationalLinearMap[
        SourceSpace: Space[Fraction, Any, int],
        TargetSpace: Space[Fraction, Any, int],
    ] = LinearMap[SourceSpace, TargetSpace]
else:
    IntegralChainComplex = RationalChainComplex = ChainComplex
    IntegralCochainComplex = RationalCochainComplex = CochainComplex
    IntegralChainSpace = RationalChainSpace = Space
    IntegralCochainSpace = RationalCochainSpace = Space
    IntegralChain = RationalChain = Element
    IntegralCochain = RationalCochain = Element
    IntegralLinearMap = RationalLinearMap = LinearMap

__all__ = [
    "IntegerRing",
    "RationalField",
    "ZZ",
    "QQ",
    "Chain",
    "Cochain",
    "BigIntEncoding",
    "ReducedFractionEncoding",
    "Space",
    "Element",
    "LinearMap",
    "ChainComplex",
    "CochainComplex",
    "ChainIsomorphism",
    "ChainError",
    "ChainLawLimit",
    "DEFAULT_LAW_LIMIT",
    "CsrEstimate",
    "CsrBuildLimit",
    "Csr",
    "IntegerCsrParts",
    "RationalCsrParts",
    "HomologyError",
    "HomologyLimit",
    "DEFAULT_HOMOLOGY_LIMIT",
    "HomologyGroup",
    "IntegralHomology",
    "analyze_integral_homology",
    "IntegralChainComplex",
    "RationalChainComplex",
    "IntegralCochainComplex",
    "RationalCochainComplex",
    "IntegralChainSpace",
    "RationalChainSpace",
    "IntegralCochainSpace",
    "RationalCochainSpace",
    "IntegralChain",
    "RationalChain",
    "IntegralCochain",
    "RationalCochain",
    "IntegralLinearMap",
    "RationalLinearMap",
]
