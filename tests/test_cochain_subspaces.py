from __future__ import annotations

import numpy as np
import pytest

from polygeo import (
    ORDINARY_FORM,
    CochainSubspace,
    Complex,
    OperatorError,
    SimplicialError,
    extend_zero,
    restrict,
    topological_boundary,
)


def _disk() -> Complex:
    return Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    )


def test_cochain_subspace_owns_strict_canonical_indices_and_parent() -> None:
    parent = _disk().cochain_space(1)
    indices = np.array([0, 2, 4], dtype=np.int64)

    subspace = CochainSubspace(parent, indices)
    indices[:] = 1
    exposed = subspace.indices()
    exposed[:] = 1

    assert subspace.parent is parent
    assert subspace.complex is parent.complex
    assert subspace.degree == 1
    assert subspace.size == 3
    np.testing.assert_array_equal(subspace.indices(), [0, 2, 4])


def test_cochain_subspace_rejects_noncanonical_indices() -> None:
    parent = _disk().cochain_space(1)
    invalid = (
        np.array([[0, 1]], dtype=np.int64),
        np.array([0.0, 1.0]),
        np.array([True, False]),
        np.array([0, 0], dtype=np.int64),
        np.array([2, 1], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        np.array([parent.size], dtype=np.int64),
    )
    for indices in invalid:
        with pytest.raises(SimplicialError):
            CochainSubspace(parent, indices)


def test_complement_is_canonical_and_subspace_identity_includes_parent() -> None:
    complex_ = _disk()
    parent = complex_.cochain_space(1)
    selected = CochainSubspace(parent, np.array([0, 2, 4], dtype=np.int64))

    complement = selected.complement()

    np.testing.assert_array_equal(complement.indices(), [1, 3])
    assert complement.parent is parent
    assert selected.same_space(
        CochainSubspace(parent, np.array([0, 2, 4], dtype=np.int64))
    )
    assert not selected.same_space(complement)

    equivalent_parent = complex_.cochain_space(1)
    equivalent = CochainSubspace(equivalent_parent, np.array([0, 2, 4], dtype=np.int64))
    assert selected.same_space(equivalent)


def test_restrict_and_extend_zero_obey_selection_laws_without_orientation_signs() -> (
    None
):
    complex_ = _disk()
    parent = complex_.cochain_space(1)
    subspace = CochainSubspace(parent, np.array([0, 2, 4], dtype=np.int64))
    restriction = restrict(parent, subspace)
    extension = extend_zero(subspace, parent)
    value = parent.form(np.array([1.0, -2.0, 3.0, -4.0, 5.0]), ORDINARY_FORM)

    restricted = restriction.apply(value)
    restored = extension.apply(restricted)

    assert restriction.source is parent
    assert restriction.target is subspace
    assert extension.source is subspace
    assert extension.target is parent
    np.testing.assert_array_equal(restricted.coefficients(), [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(restored.coefficients(), [1.0, 0.0, 3.0, 0.0, 5.0])
    np.testing.assert_array_equal(
        (restriction.matrix() @ extension.matrix()).toarray(), np.eye(3)
    )
    np.testing.assert_array_equal(
        (extension.matrix() @ restriction.matrix()).toarray(),
        np.diag([1.0, 0.0, 1.0, 0.0, 1.0]),
    )


def test_empty_subspace_has_valid_zero_sized_transfer_laws() -> None:
    parent = _disk().cochain_space(1)
    empty = CochainSubspace(parent, np.array([], dtype=np.int64))
    restriction = restrict(parent, empty)
    extension = extend_zero(empty, parent)

    assert restriction.matrix().shape == (0, parent.size)
    assert extension.matrix().shape == (parent.size, 0)
    assert (restriction.matrix() @ extension.matrix()).shape == (0, 0)
    assert (extension.matrix() @ restriction.matrix()).nnz == 0
    assert (
        empty.form(np.array([], dtype=np.float64), ORDINARY_FORM).coefficients().size
        == 0
    )
    np.testing.assert_array_equal(empty.complement().indices(), np.arange(parent.size))


def test_transfer_maps_require_the_exact_parent_object() -> None:
    complex_ = _disk()
    retained_parent = complex_.cochain_space(1)
    equivalent_parent = complex_.cochain_space(1)
    subspace = CochainSubspace(retained_parent, np.array([0], dtype=np.int64))

    with pytest.raises(OperatorError, match="exact parent"):
        restrict(equivalent_parent, subspace)
    with pytest.raises(OperatorError, match="exact parent"):
        extend_zero(subspace, equivalent_parent)


def test_boundary_subspace_uses_canonical_parent_basis_indices() -> None:
    domain = _disk().boundary_regular()
    boundary = topological_boundary(domain)
    parent = domain.cochain_space(1)
    indices = np.flatnonzero(boundary.mask(1)).astype(np.int64)
    subspace = CochainSubspace(parent, indices)
    value = parent.form(np.arange(parent.size, dtype=np.float64), ORDINARY_FORM)

    restricted = restrict(parent, subspace).apply(value)

    np.testing.assert_array_equal(subspace.indices(), [0, 2, 3, 4])
    np.testing.assert_array_equal(restricted.coefficients(), [0.0, 2.0, 3.0, 4.0])
