# ty-expect: invalid-argument-type, invalid-argument-type, unresolved-attribute, invalid-argument-type, invalid-argument-type, invalid-argument-type, missing-argument, invalid-argument-type

from polygeo.chain import (
    HomologyGroup,
    IntegralChainComplex,
    QQ,
    analyze_integral_homology,
)


def invalid_exact_relations(complex_: IntegralChainComplex) -> None:
    chain = complex_[1].element({0: 1})
    cochain = complex_.dual()[1].element({0: 1})
    complex_.boundary(1).apply(cochain)
    complex_.dual().coboundary(1).apply(chain)
    chain.dual()
    complex_.over(QQ)[1].element({0: 1.5})
    analyze_integral_homology(complex_.over(QQ), [0])
    analyze_integral_homology(complex_, ["0"])
    HomologyGroup()
    cochain.cup(chain)
