"""Exact fixed-band arithmetic for admitted binary64 values."""

from __future__ import annotations

from collections.abc import Iterable


BINARY64_PRODUCT_LATTICE_BITS = 2148
BINARY64_PRODUCT_LATTICE_DENOMINATOR = 1 << BINARY64_PRODUCT_LATTICE_BITS


def binary64_lattice(value: float) -> int:
    """Represent one finite binary64 value on the product lattice."""
    numerator, denominator = value.as_integer_ratio()
    return numerator << (BINARY64_PRODUCT_LATTICE_BITS - (denominator.bit_length() - 1))


def binary64_ratio(value: float) -> tuple[int, int]:
    """Return an exact numerator and power-of-two denominator exponent."""
    numerator, denominator = value.as_integer_ratio()
    return numerator, denominator.bit_length() - 1


def binary64_sum_product_lattice(
    left: Iterable[float],
    right: Iterable[float],
) -> int:
    """Return the exact sum of binary64 products on the product lattice."""
    exact = 0
    for left_value, right_value in zip(left, right, strict=True):
        if left_value == 0.0 or right_value == 0.0:
            continue
        left_numerator, left_denominator_bits = binary64_ratio(left_value)
        right_numerator, right_denominator_bits = binary64_ratio(right_value)
        exact += (left_numerator * right_numerator) << (
            BINARY64_PRODUCT_LATTICE_BITS
            - left_denominator_bits
            - right_denominator_bits
        )
    return exact


def binary64_from_lattice(exact: int) -> float:
    """Round one product-lattice integer to binary64."""
    return exact / BINARY64_PRODUCT_LATTICE_DENOMINATOR
