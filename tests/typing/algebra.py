from fractions import Fraction
from typing import Literal, assert_type

from polygeo import (
    BigIntEncoding,
    Complex,
    CsrRepresentation,
    HomologyGroup,
    IntegralChain,
    IntegralChainComplex,
    IntegralChainSpace,
    IntegralCochain,
    IntegralCochainSpace,
    IntegralHomology,
    QQ,
    RationalChain,
    prepare_integral_homology,
)


def exact_relations(complex_: IntegralChainComplex) -> None:
    chains = complex_[1]
    cochains = complex_.dual()[1]
    chain = chains.element({0: 1})
    cochain = cochains.element({0: 2})
    assert_type(chains, IntegralChainSpace[int])
    assert_type(cochains, IntegralCochainSpace[int])
    assert_type(chain, IntegralChain[int])
    assert_type(cochain, IntegralCochain[int])
    assert_type(cochain.evaluate(chain), int)

    boundary = complex_.boundary(1)
    assert_type(boundary.apply(chain), IntegralChain[int])
    assert_type(
        boundary.dual().apply(complex_.dual()[0].element({0: 1})),
        IntegralCochain[int],
    )
    estimate = CsrRepresentation.estimate(boundary, BigIntEncoding)
    representation = CsrRepresentation.build(
        boundary, BigIntEncoding, estimate.as_limit()
    )
    assert_type(representation.apply(chain), IntegralChain[int])

    rational = complex_.over(QQ)[1].element({0: Fraction(1, 3)})
    assert_type(rational, RationalChain[int])


def homology_relations(complex_: Complex) -> None:
    analysis = prepare_integral_homology(complex_.chain_complex(), [0, 1])
    assert_type(analysis, IntegralHomology)
    group = analysis[1]
    assert_type(group, HomologyGroup[Literal[1]])
    assert_type(group.free_cycle(0), IntegralChain[Literal[1]])
