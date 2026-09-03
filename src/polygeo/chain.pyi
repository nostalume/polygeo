from __future__ import annotations
from fractions import Fraction
from typing import Any
from ._polygeo_native import (
    BigIntEncoding as BigIntEncoding,
    Chain as Chain,
    ChainComplex as ChainComplex,
    ChainError as ChainError,
    ChainIsomorphism as ChainIsomorphism,
    ChainLawLimit as ChainLawLimit,
    Cochain as Cochain,
    CochainComplex as CochainComplex,
    Csr as Csr,
    CsrBuildLimit as CsrBuildLimit,
    CsrEstimate as CsrEstimate,
    DEFAULT_HOMOLOGY_LIMIT as DEFAULT_HOMOLOGY_LIMIT,
    DEFAULT_LAW_LIMIT as DEFAULT_LAW_LIMIT,
    Element as Element,
    HomologyError as HomologyError,
    HomologyGroup as HomologyGroup,
    HomologyLimit as HomologyLimit,
    IntegerCsrParts as IntegerCsrParts,
    IntegerRing as IntegerRing,
    IntegralHomology as IntegralHomology,
    LinearMap as LinearMap,
    QQ as QQ,
    RationalCsrParts as RationalCsrParts,
    RationalField as RationalField,
    ReducedFractionEncoding as ReducedFractionEncoding,
    Space as Space,
    ZZ as ZZ,
    analyze_integral_homology as analyze_integral_homology,
)

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
type IntegralLinearMap[S: Space[int, Any, int], T: Space[int, Any, int]] = LinearMap[
    S, T
]
type RationalLinearMap[S: Space[Fraction, Any, int], T: Space[Fraction, Any, int]] = (
    LinearMap[S, T]
)

__all__: list[str]
