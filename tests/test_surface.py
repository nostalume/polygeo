from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import polygeo
import pytest

from polygeo import (
    Complex,
    Disk,
    Geometry,
    SimplicialError,
    SurfaceError,
    disk,
    gaussian_curvature_measure,
)


_ROOT_EXPORTS = {
    "ORDINARY_FORM",
    "AlgorithmError",
    "AssembledSystem",
    "BasisCoordinates",
    "BoundaryState",
    "BoundaryUnknown",
    "Certified",
    "CochainSpace",
    "CochainSubspace",
    "CodimensionOneRegular",
    "Complex",
    "ConditionEvidence",
    "Connected",
    "ConnectivityState",
    "ConnectivityUnknown",
    "DirichletProblem",
    "DirectionFieldEvidence",
    "Disk",
    "DualCochainSpace",
    "FaceDirectionField",
    "FaceVectors",
    "FieldSemantics",
    "Form",
    "FrozenFlowEvidence",
    "Geometry",
    "GeometryError",
    "HodgeComponents",
    "HodgeEvidence",
    "HolonomyEvidence",
    "IntegralDualCycles",
    "IntegrableConnection",
    "LeastSquaresSolution",
    "LinearMap",
    "LinearSolution",
    "MeanZeroProblem",
    "MeshError",
    "NumericalError",
    "OneForm",
    "OperatorError",
    "OrdinaryForm",
    "PlotError",
    "OrientationState",
    "OrientationUnknown",
    "Oriented",
    "PositiveHodgeMetric",
    "PrepareLeastSquares",
    "PrepareLinearSolve",
    "PreparedLeastSquares",
    "PreparedLinearSolve",
    "RealHomologyBasis",
    "ResidualEvidence",
    "SimplexSubset",
    "Simplicial",
    "SimplicialError",
    "SurfaceError",
    "SurfaceConnection",
    "SystemError",
    "TopologyState",
    "TriangleFrames",
    "TriangleManifold",
    "TwoForm",
    "VertexMap",
    "VertexVectors",
    "WithBoundary",
    "WithoutBoundary",
    "ZeroForm",
    "admit_integrable_connection",
    "assemble_poisson",
    "codifferential",
    "connection_holonomy",
    "disk",
    "face_unit_normals",
    "eliminate_dirichlet",
    "extend_zero",
    "exterior_derivative",
    "gaussian_curvature_measure",
    "integral_dual_cycles",
    "integrate_direction_field",
    "levi_civita_connection",
    "harmonic_extension",
    "hodge_decomposition",
    "hodge_laplacian",
    "hodge_star",
    "impose_mean_zero",
    "load_surface",
    "mean_curvature_flow_step",
    "mean_curvature_vectors",
    "prepare_direct",
    "prepare_least_squares",
    "plot_cochain",
    "plot_geometry",
    "plot_homology_cycle",
    "plot_surface_vectors",
    "real_homology_basis",
    "restrict",
    "sphere_inscribed_vertex_normals",
    "surface_area_gradient",
    "surface_connection",
    "tip_angle_vertex_normals",
    "topological_boundary",
    "triangle_frames",
    "uniform_vertex_normals",
    "volume_gradient",
    "vertex_map",
    "weighted_pairing",
}


def test_root_public_boundary_is_exact() -> None:
    assert len(polygeo.__all__) == len(set(polygeo.__all__))
    assert set(polygeo.__all__) == _ROOT_EXPORTS
    assert all(hasattr(polygeo, name) for name in _ROOT_EXPORTS)
    assert {
        "CircumcentricDual",
        "HarmonicBasis",
        "SO2Connection",
        "SurfaceOneForm",
    }.isdisjoint(polygeo.__all__)


def _triangle_disk(*, oriented: bool = True):
    domain = (
        Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
        .triangle_manifold()
        .with_boundary()
        .connected()
    )
    return domain.oriented() if oriented else domain


def _tetrahedron(scale: float = 1.0, shift: float = 0.0) -> Geometry:
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)
    domain = (
        Complex.from_maximal_simplices(faces).triangle_manifold().without_boundary()
    )
    positions = shift + scale * np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=np.float64,
    )
    return Geometry.from_positions(domain, positions)


def _saddle_disk(scale: float = 1.0) -> Geometry:
    count = 8
    angles = 2.0 * math.pi * np.arange(count) / count
    ring = np.column_stack(
        (np.cos(angles), np.sin(angles), 0.7 * (-1.0) ** np.arange(count))
    )
    positions = scale * np.vstack((np.zeros(3), ring))
    faces = np.array(
        [[0, 1 + index, 1 + (index + 1) % count] for index in range(count)],
        dtype=np.int64,
    )
    domain = Complex.from_maximal_simplices(faces).triangle_manifold().with_boundary()
    return Geometry.from_positions(domain, positions)


def _annulus():
    count = 6
    angles = 2.0 * math.pi * np.arange(count) / count
    positions = np.vstack(
        (
            np.column_stack((np.cos(angles), np.sin(angles))),
            2.1 * np.column_stack((np.cos(angles + 0.05), np.sin(angles + 0.05))),
        )
    )
    faces: list[tuple[int, int, int]] = []
    for index in range(count):
        inner = index
        next_inner = (index + 1) % count
        outer = count + index
        next_outer = count + (index + 1) % count
        faces.extend(((inner, outer, next_inner), (next_inner, outer, next_outer)))
    domain = (
        Complex.from_maximal_simplices(np.array(faces, dtype=np.int64))
        .triangle_manifold()
        .oriented()
        .with_boundary()
        .connected()
    )
    return domain, Geometry.from_positions(domain, positions)


def _punctured_torus():
    major_count, minor_count = 8, 6

    def index(major: int, minor: int) -> int:
        return (major % major_count) * minor_count + minor % minor_count

    faces: list[tuple[int, int, int]] = []
    for major in range(major_count):
        for minor in range(minor_count):
            a, b = index(major, minor), index(major + 1, minor)
            c, d = index(major + 1, minor + 1), index(major, minor + 1)
            faces.extend(((a, b, c), (a, c, d)))
    faces.pop(0)
    return (
        Complex.from_maximal_simplices(np.array(faces, dtype=np.int64))
        .triangle_manifold()
        .oriented()
        .with_boundary()
        .connected()
    )


def test_disk_binds_exact_complex_and_rejects_public_construction() -> None:
    domain = _triangle_disk()
    evidence = disk(domain)

    assert isinstance(evidence, Disk)
    assert evidence.complex is domain
    with pytest.raises(AttributeError):
        evidence._complex = _triangle_disk()
    with pytest.raises(SurfaceError, match="created by disk"):
        Disk()


def test_disk_rejects_annulus_and_disconnected_disks() -> None:
    annulus, _ = _annulus()
    with pytest.raises(SurfaceError, match="Euler characteristic|closed component"):
        disk(annulus)

    disconnected = (
        Complex.from_maximal_simplices(np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64))
        .triangle_manifold()
        .oriented()
        .with_boundary()
    )
    with pytest.raises(SurfaceError, match="connected oriented triangle manifold"):
        disk(cast(Any, disconnected))

    with pytest.raises(SurfaceError, match="Euler characteristic"):
        disk(_punctured_torus())


def test_disk_prerequisites_reject_pinched_and_nonmanifold_inputs() -> None:
    isolated = Complex.from_maximal_simplices(
        np.array([[0, 1, 2]], dtype=np.int64), vertex_count=4
    )
    with pytest.raises(SimplicialError):
        isolated.triangle_manifold()

    pinched = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    )
    with pytest.raises(SimplicialError):
        pinched.triangle_manifold()

    fan = Complex.from_maximal_simplices(
        np.array([[0, 1, 2], [0, 2, 3], [0, 2, 4]], dtype=np.int64)
    )
    with pytest.raises(SimplicialError):
        fan.triangle_manifold()


def test_curvature_closed_gauss_bonnet_identity_and_ownership() -> None:
    geometry = _tetrahedron()
    curvature = gaussian_curvature_measure(geometry)

    assert curvature.space.complex is geometry.complex
    np.testing.assert_allclose(curvature.coefficients(), np.full(4, math.pi))
    assert math.fsum(curvature.coefficients()) == pytest.approx(4.0 * math.pi)
    exposed = curvature.coefficients()
    exposed[:] = 0.0
    np.testing.assert_allclose(curvature.coefficients(), np.full(4, math.pi))


def test_curvature_boundary_defects_mixed_sign_and_gauss_bonnet() -> None:
    geometry = _saddle_disk()
    curvature = gaussian_curvature_measure(geometry).coefficients()

    assert curvature[0] < 0.0
    assert np.all(curvature[1:] > 0.0)
    assert math.fsum(curvature) == pytest.approx(2.0 * math.pi)

    _, annulus_geometry = _annulus()
    assert math.fsum(
        gaussian_curvature_measure(annulus_geometry).coefficients()
    ) == pytest.approx(0.0, abs=3.0e-14)


def test_curvature_is_orientation_free_and_additive_by_component() -> None:
    segment_count = 8
    faces: list[tuple[int, int, int]] = []
    positions = []
    for index in range(segment_count):
        left, right = 2 * index, 2 * index + 1
        next_left, next_right = (
            (2 * (index + 1), 2 * (index + 1) + 1)
            if index + 1 < segment_count
            else (1, 0)
        )
        faces.extend(((left, next_left, next_right), (left, next_right, right)))
        angle = 2.0 * math.pi * index / segment_count
        for offset in (-0.3, 0.3):
            positions.append(
                (
                    (2.0 + offset * math.cos(angle / 2.0)) * math.cos(angle),
                    (2.0 + offset * math.cos(angle / 2.0)) * math.sin(angle),
                    offset * math.sin(angle / 2.0),
                )
            )
    mobius = (
        Complex.from_maximal_simplices(np.array(faces, dtype=np.int64))
        .triangle_manifold()
        .with_boundary()
        .connected()
    )
    with pytest.raises(SimplicialError):
        mobius.oriented()
    mobius_geometry = Geometry.from_positions(mobius, np.array(positions))
    assert math.fsum(
        gaussian_curvature_measure(mobius_geometry).coefficients()
    ) == pytest.approx(0.0, abs=4.0e-14)

    combined_faces = np.array(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3], [4, 5, 6]],
        dtype=np.int64,
    )
    combined = Complex.from_maximal_simplices(combined_faces).triangle_manifold()
    combined_geometry = Geometry.from_positions(
        combined,
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
            ]
        ),
    )
    assert math.fsum(
        gaussian_curvature_measure(combined_geometry).coefficients()
    ) == pytest.approx(6.0 * math.pi)


def test_curvature_is_orientation_free_scale_and_rigid_motion_invariant() -> None:
    baseline = gaussian_curvature_measure(_tetrahedron()).coefficients()
    for scale, shift in ((1.0e-100, 0.0), (1.0e100, 0.0), (1.0, 1.0e12)):
        np.testing.assert_allclose(
            gaussian_curvature_measure(_tetrahedron(scale, shift)).coefficients(),
            baseline,
            rtol=4.0e-15,
            atol=4.0e-15,
        )


def test_curvature_handles_represented_skinny_angles_and_numpy_policy() -> None:
    domain = _triangle_disk(oriented=False)
    geometry = Geometry.from_positions(
        domain,
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0e-300]], dtype=np.float64),
    )
    previous = np.seterr(all="raise")
    try:
        expected = np.geterr().copy()
        values = gaussian_curvature_measure(geometry).coefficients()
        assert np.all(np.isfinite(values))
        assert math.fsum(values) == pytest.approx(2.0 * math.pi)
        assert np.geterr() == expected
    finally:
        np.seterr(**previous)


def test_curvature_closes_qr_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = _tetrahedron()

    def fail(*_args: object, **_kwargs: object) -> None:
        raise np.linalg.LinAlgError("private backend detail")

    monkeypatch.setattr(np.linalg, "qr", fail)
    with pytest.raises(SurfaceError, match="angle evaluation failed") as caught:
        gaussian_curvature_measure(geometry)
    assert isinstance(caught.value.__cause__, np.linalg.LinAlgError)
    assert "private backend detail" not in str(caught.value)


def test_curvature_rejects_nontriangle_geometry() -> None:
    raw = Complex.from_maximal_simplices(np.array([[0, 1]], dtype=np.int64))
    geometry = Geometry.from_positions(raw, np.array([[0.0], [1.0]], dtype=np.float64))
    with pytest.raises(SurfaceError, match="triangle-manifold geometry"):
        gaussian_curvature_measure(cast(Any, geometry))
