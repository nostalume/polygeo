from typing import Literal
from ._polygeo_native import (
    FormElement as Element,
    FormElementError as ElementError,
    FormOperator as Operator,
    FormSpace as Space,
    OperatorError as OperatorError,
)

type ChainSpace[Degree: int] = Space[Literal["chain"], Degree]
type CochainSpace[Degree: int] = Space[Literal["cochain"], Degree]
type Chain[Degree: int] = Element[Literal["chain"], Degree]
type Cochain[Degree: int] = Element[Literal["cochain"], Degree]

__all__ = [
    "Space",
    "Element",
    "Operator",
    "ElementError",
    "OperatorError",
    "ChainSpace",
    "CochainSpace",
    "Chain",
    "Cochain",
]
