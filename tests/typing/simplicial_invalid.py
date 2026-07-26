# ty-expect: invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type, invalid-argument-type

from typing import Literal

import numpy as np

from polygeo import (
    ORDINARY_FORM,
    CochainSpace,
    WithoutBoundary,
    Complex,
    Connected,
    FieldSemantics,
    Form,
    OrdinaryForm,
    Oriented,
    TriangleManifold,
)

qualified = (
    Complex.from_maximal_simplices(
        np.array(
            [
                [1, 2, 3],
                [0, 3, 2],
                [0, 1, 3],
                [0, 2, 1],
            ],
            dtype=np.int64,
        )
    )
    .triangle_manifold()
    .oriented()
    .without_boundary()
    .connected()
)

qualified.triangle_manifold()
qualified.oriented()
qualified.without_boundary()
qualified.connected()

unrefined = Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
unrefined.without_boundary()
unrefined.with_boundary()


type Domain = Complex[WithoutBoundary, Oriented, Connected, TriangleManifold]


class AlternateSemantics(FieldSemantics):
    pass


def consume(
    value: Form[CochainSpace[Domain, Literal[1]], OrdinaryForm],
) -> None:
    pass


zero = qualified.cochain_space(0).form(
    np.zeros(qualified.simplex_count(0)), ORDINARY_FORM
)
alternate = qualified.cochain_space(1).form(
    np.zeros(qualified.simplex_count(1)), AlternateSemantics()
)
consume(zero)
consume(alternate)
