use std::f64::consts::PI;
use std::num::NonZeroU32;
use std::sync::Arc;

use num_bigint::BigInt;
use polygeo_core::{
    Binary64CochainSpace, Binary64Element, CancellationToken, CandidateInput, ComplexCore,
    EuclideanRealization, HomologyLimit, IntegralHomology, NativeExecutor, NondegenerateCapability,
    PairingCapability, RealizationLimit, SolveError, StorageLimit, SurfaceError, TriangleSurface,
    WorkLimit,
};

fn tetrahedron(scale: f64, translation: [f64; 3]) -> Arc<EuclideanRealization> {
    let topology = ComplexCore::admit(
        CandidateInput::unsigned([0_u64, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3], 4, 3, Some(4)).unwrap(),
    )
    .unwrap();
    let base = [
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ];
    let positions = base
        .into_iter()
        .flat_map(|point| {
            point
                .into_iter()
                .zip(translation)
                .map(move |(coordinate, offset)| scale * coordinate + offset)
        })
        .collect();
    EuclideanRealization::admit(topology, 3, positions, RealizationLimit::DEFAULT).unwrap()
}

fn octahedron() -> Arc<EuclideanRealization> {
    let topology = ComplexCore::admit(
        CandidateInput::unsigned(
            [
                4_u64, 0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 5, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3,
            ],
            8,
            3,
            Some(6),
        )
        .unwrap(),
    )
    .unwrap();
    EuclideanRealization::admit(
        topology,
        3,
        vec![
            1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            -1.0,
        ],
        RealizationLimit::DEFAULT,
    )
    .unwrap()
}

fn triangle() -> Arc<EuclideanRealization> {
    let topology =
        ComplexCore::admit(CandidateInput::unsigned([0_u64, 1, 2], 1, 3, Some(3)).unwrap())
            .unwrap();
    EuclideanRealization::admit(
        topology,
        3,
        vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        RealizationLimit::DEFAULT,
    )
    .unwrap()
}

fn nonplanar_disk() -> Arc<EuclideanRealization> {
    let topology = ComplexCore::admit(
        CandidateInput::unsigned([0_u64, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], 4, 3, Some(5)).unwrap(),
    )
    .unwrap();
    EuclideanRealization::admit(
        topology,
        3,
        vec![
            -1.0, -1.0, 0.0, 1.0, -1.0, 0.2, 1.0, 1.0, 0.0, -1.0, 1.0, -0.1, 0.0, 0.0, 0.5,
        ],
        RealizationLimit::DEFAULT,
    )
    .unwrap()
}

fn torus(major_sections: usize, minor_sections: usize) -> Arc<EuclideanRealization> {
    let mut positions = Vec::with_capacity(3 * major_sections * minor_sections);
    for major in 0..major_sections {
        let theta = 2.0 * PI * f64::from(u32::try_from(major).unwrap())
            / f64::from(u32::try_from(major_sections).unwrap());
        for minor in 0..minor_sections {
            let phi = 2.0 * PI * (f64::from(u32::try_from(minor).unwrap()) + 0.1 * theta.sin())
                / f64::from(u32::try_from(minor_sections).unwrap());
            let radius = 2.0 + phi.cos();
            positions.extend([radius * theta.cos(), radius * theta.sin(), phi.sin()]);
        }
    }
    let mut faces = Vec::new();
    for major in 0..major_sections {
        for minor in 0..minor_sections {
            let lower = major * minor_sections + minor;
            let major_next = ((major + 1) % major_sections) * minor_sections + minor;
            let diagonal =
                ((major + 1) % major_sections) * minor_sections + (minor + 1) % minor_sections;
            let minor_next = major * minor_sections + (minor + 1) % minor_sections;
            let point = |vertex: usize| -> [f64; 3] {
                positions[3 * vertex..3 * vertex + 3].try_into().unwrap()
            };
            let first_weight = cotangent(point(lower), point(diagonal), point(major_next))
                + cotangent(point(lower), point(diagonal), point(minor_next));
            let second_weight = cotangent(point(major_next), point(minor_next), point(lower))
                + cotangent(point(major_next), point(minor_next), point(diagonal));
            if first_weight >= second_weight {
                faces.extend([[lower, major_next, diagonal], [lower, diagonal, minor_next]]);
            } else {
                faces.extend([
                    [lower, major_next, minor_next],
                    [major_next, diagonal, minor_next],
                ]);
            }
        }
    }
    let topology = ComplexCore::admit(
        CandidateInput::unsigned(
            faces
                .iter()
                .flatten()
                .map(|value| u64::try_from(*value).unwrap()),
            faces.len(),
            3,
            Some(major_sections * minor_sections),
        )
        .unwrap(),
    )
    .unwrap();
    EuclideanRealization::admit(topology, 3, positions, RealizationLimit::DEFAULT).unwrap()
}

fn cotangent(left: [f64; 3], right: [f64; 3], opposite: [f64; 3]) -> f64 {
    let left = std::array::from_fn::<_, 3, _>(|axis| left[axis] - opposite[axis]);
    let right = std::array::from_fn::<_, 3, _>(|axis| right[axis] - opposite[axis]);
    let dot = left
        .iter()
        .zip(right)
        .map(|(&left, right)| left * right)
        .sum::<f64>();
    let cross = [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ];
    dot / norm(&cross)
}

fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn assert_close(left: f64, right: f64, tolerance: f64) {
    assert!((left - right).abs() <= tolerance, "{left} != {right}");
}

fn assert_lscm_failures(surface: &TriangleSurface) {
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert_eq!(
        surface
            .least_squares_conformal_map(
                [0, 2],
                RealizationLimit::DEFAULT,
                &NativeExecutor::sequential(),
                StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
                WorkLimit::new(u64::MAX),
                &cancellation,
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::Cancelled)
    );
    assert_eq!(
        surface
            .least_squares_conformal_map(
                [0, 2],
                RealizationLimit::DEFAULT,
                &NativeExecutor::sequential(),
                StorageLimit::new(0, 0).unwrap(),
                WorkLimit::new(u64::MAX),
                &CancellationToken::new(),
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::ResourceLimit)
    );
    assert_eq!(
        surface
            .least_squares_conformal_map(
                [0, 4],
                RealizationLimit::DEFAULT,
                &NativeExecutor::sequential(),
                StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
                WorkLimit::new(u64::MAX),
                &CancellationToken::new(),
            )
            .unwrap_err()
            .surface(),
        Some(SurfaceError::AnchorNotBoundary)
    );
}

#[test]
fn least_squares_conformal_map_preserves_explicit_anchors_and_certifies_the_disk() {
    let surface = TriangleSurface::admit(nonplanar_disk()).unwrap();
    let solution = surface
        .least_squares_conformal_map(
            [0, 2],
            RealizationLimit::DEFAULT,
            &NativeExecutor::sequential(),
            StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
            WorkLimit::new(u64::MAX),
            &CancellationToken::new(),
        )
        .unwrap();

    assert!(Arc::ptr_eq(
        solution.realization().topology(),
        surface.realization().topology()
    ));
    assert_eq!(solution.realization().ambient_dimension(), 2);
    let positions = solution.realization().positions();
    assert_eq!(&positions[0..2], &[0.0, 0.0]);
    assert_eq!(&positions[4..6], &[1.0, 0.0]);

    let evidence = solution.evidence();
    assert_eq!(evidence.required_rank(), 6);
    assert_eq!(evidence.observed_rank(), 6);
    assert!(evidence.condition_indicator().is_finite());
    assert!(evidence.condition_indicator() >= 1.0);
    assert!(evidence.residual_bound() < 1.0);
    assert!(evidence.minimum_normalized_signed_twice_area() > 0.0);

    let mapped = positions.to_vec();
    let transformed_positions = surface
        .realization()
        .positions()
        .chunks_exact(3)
        .flat_map(|point| {
            [
                -2.0 * point[1] + 5.0,
                2.0 * point[0] - 7.0,
                2.0 * point[2] + 3.0,
            ]
        })
        .collect();
    let transformed = EuclideanRealization::admit(
        Arc::clone(surface.realization().topology()),
        3,
        transformed_positions,
        RealizationLimit::DEFAULT,
    )
    .unwrap();
    let transformed = TriangleSurface::admit(transformed)
        .unwrap()
        .least_squares_conformal_map(
            [0, 2],
            RealizationLimit::DEFAULT,
            &NativeExecutor::sequential(),
            StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
            WorkLimit::new(u64::MAX),
            &CancellationToken::new(),
        )
        .unwrap();
    for (&left, &right) in mapped.iter().zip(transformed.realization().positions()) {
        assert_close(left, right, 2.0e-13);
    }

    let reversed = surface
        .least_squares_conformal_map(
            [2, 0],
            RealizationLimit::DEFAULT,
            &NativeExecutor::sequential(),
            StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
            WorkLimit::new(u64::MAX),
            &CancellationToken::new(),
        )
        .unwrap();
    for (forward, reverse) in mapped
        .chunks_exact(2)
        .zip(reversed.realization().positions().chunks_exact(2))
    {
        assert_close(reverse[0], 1.0 - forward[0], 2.0e-13);
        assert_close(reverse[1], -forward[1], 2.0e-13);
    }

    assert_lscm_failures(&surface);
}

fn assert_triangle_differential(
    realization: &Arc<EuclideanRealization>,
    scalar: [f64; 3],
    vector: [f64; 3],
    expected_gradient: [f64; 3],
    expected_divergence: [f64; 3],
) {
    let surface = TriangleSurface::admit(Arc::clone(realization)).unwrap();
    let scalar = Binary64Element::admit(
        Binary64CochainSpace::full(Arc::clone(realization.topology()), 0).unwrap(),
        scalar.to_vec(),
    )
    .unwrap();
    let gradient = surface.gradient(&scalar).unwrap();
    for (&actual, expected) in gradient.values().iter().zip(expected_gradient) {
        assert_close(actual, expected, 2.0e-14);
    }
    let field = surface.face_vectors(vector.to_vec()).unwrap();
    let divergence = surface.divergence(&field).unwrap();
    for (&actual, expected) in divergence.coefficients().iter().zip(expected_divergence) {
        assert_close(actual, expected, 2.0e-14);
    }
}

fn assert_differential_rejections(
    surface: &TriangleSurface,
    realization: &Arc<EuclideanRealization>,
    scalar: &[f64],
    vector: &[f64],
) {
    let foreign = TriangleSurface::admit(triangle()).unwrap();
    let foreign_scalar = Binary64Element::admit(
        Binary64CochainSpace::full(Arc::clone(foreign.realization().topology()), 0).unwrap(),
        scalar.to_vec(),
    )
    .unwrap();
    assert_eq!(
        surface.gradient(&foreign_scalar).unwrap_err(),
        SurfaceError::OwnerMismatch
    );

    for invalid in [
        Binary64Element::admit(
            Binary64CochainSpace::full(Arc::clone(realization.topology()), 1).unwrap(),
            vec![0.0; 3],
        )
        .unwrap(),
        Binary64Element::admit(
            Binary64CochainSpace::selected(Arc::new(
                realization.topology().selection(0, vec![0, 1]).unwrap(),
            ))
            .unwrap(),
            vec![1.0, 7.0],
        )
        .unwrap(),
    ] {
        assert_eq!(
            surface.gradient(&invalid).unwrap_err(),
            SurfaceError::OwnerMismatch
        );
    }
    assert_eq!(
        surface
            .divergence(&foreign.face_vectors(vector.to_vec()).unwrap())
            .unwrap_err(),
        SurfaceError::OwnerMismatch
    );
    assert_eq!(
        surface
            .divergence(&surface.vertex_vectors(vec![0.0; 9]).unwrap())
            .unwrap_err(),
        SurfaceError::FieldShape
    );
}

fn assert_gradient_divergence_matches_stiffness() {
    let realization = tetrahedron(1.0, [0.0; 3]);
    let metric = realization
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    let surface = TriangleSurface::admit(Arc::clone(&realization)).unwrap();
    let scalar = Binary64Element::admit(
        Binary64CochainSpace::full(Arc::clone(realization.topology()), 0).unwrap(),
        vec![1.0, 7.0, -1.0, 2.0],
    )
    .unwrap();
    let load = surface
        .divergence(&surface.gradient(&scalar).unwrap())
        .unwrap();
    assert_close(load.coefficients().iter().sum(), 0.0, 2.0e-13);
    let laplacian = metric.laplacian(0).unwrap().apply(&scalar).unwrap();
    for ((&actual, &mass), &value) in load
        .coefficients()
        .iter()
        .zip(metric.hodge_coefficients_slice(0).unwrap())
        .zip(laplacian.coefficients())
    {
        assert_close(actual, -mass * value, 2.0e-13);
    }
}

#[test]
fn one_field_carrier_obeys_support_shape_and_normalization() {
    let surface = TriangleSurface::admit(tetrahedron(1.0, [0.0; 3])).unwrap();
    let normals = surface.face_unit_normals().unwrap();

    assert!(Arc::ptr_eq(normals.realization(), surface.realization()));
    assert_eq!(normals.entity_count(), 4);
    assert_eq!(normals.fiber_dimension(), 3);
    assert_eq!(normals.values().len(), 12);
    for normal in normals.values().chunks_exact(3) {
        assert_close(norm(normal), 1.0, 4.0e-15);
    }

    let area = surface.surface_area_gradient().unwrap();
    assert_eq!(area.entity_count(), 4);
    for axis in 0..3 {
        assert_close(
            area.values().chunks_exact(3).map(|row| row[axis]).sum(),
            0.0,
            4.0e-15,
        );
    }
    assert_eq!(area.normalized().unwrap().values().len(), 12);

    assert_eq!(
        surface.vertex_vectors(vec![0.0; 11]).unwrap_err(),
        SurfaceError::FieldShape
    );
    let mut invalid = vec![0.0; 12];
    invalid[0] = f64::NAN;
    assert_eq!(
        surface.face_vectors(invalid).unwrap_err(),
        SurfaceError::NonFinite
    );
}

#[test]
fn gradient_and_divergence_are_affine_negative_adjoints() {
    let realization = triangle();
    let surface = TriangleSurface::admit(Arc::clone(&realization)).unwrap();
    let scalar_space = Binary64CochainSpace::full(Arc::clone(realization.topology()), 0).unwrap();
    let scalar = Binary64Element::admit(scalar_space.clone(), vec![1.0, 7.0, -1.0]).unwrap();

    let gradient = surface.gradient(&scalar).unwrap();
    assert_eq!(gradient.values(), &[3.0, -2.0, 0.0]);
    let shifted =
        Binary64Element::admit(scalar_space, vec![1.0e12 + 1.0, 1.0e12 + 7.0, 1.0e12 - 1.0])
            .unwrap();
    assert_eq!(
        surface.gradient(&shifted).unwrap().values(),
        gradient.values()
    );
    let constant = Binary64Element::admit(
        Binary64CochainSpace::full(Arc::clone(realization.topology()), 0).unwrap(),
        vec![1.0e300; 3],
    )
    .unwrap();
    assert_eq!(surface.gradient(&constant).unwrap().values(), &[0.0; 3]);

    let field = surface.face_vectors(vec![4.0, -3.0, 7.0]).unwrap();
    let divergence = surface.divergence(&field).unwrap();
    assert_eq!(divergence.coefficients(), &[-1.0, -2.0, 3.0]);
    let pairing = scalar
        .coefficients()
        .iter()
        .zip(divergence.coefficients())
        .map(|(&value, &load)| value * load)
        .sum::<f64>();
    assert_close(pairing, -18.0, 2.0e-14);

    let scaled_realization = EuclideanRealization::admit(
        Arc::clone(realization.topology()),
        3,
        realization
            .positions()
            .iter()
            .map(|value| 3.0 * value)
            .collect(),
        RealizationLimit::DEFAULT,
    )
    .unwrap();
    assert_triangle_differential(
        &scaled_realization,
        [1.0, 7.0, -1.0],
        [4.0, -3.0, 7.0],
        [1.0, -2.0 / 3.0, 0.0],
        [-3.0, -6.0, 9.0],
    );

    let rotated_realization = EuclideanRealization::admit(
        Arc::clone(realization.topology()),
        3,
        realization
            .positions()
            .chunks_exact(3)
            .flat_map(|point| [-point[1] + 5.0, point[0] - 7.0, point[2] + 11.0])
            .collect(),
        RealizationLimit::DEFAULT,
    )
    .unwrap();
    assert_triangle_differential(
        &rotated_realization,
        [1.0, 7.0, -1.0],
        [3.0, 4.0, 7.0],
        [2.0, 3.0, 0.0],
        [-1.0, -2.0, 3.0],
    );

    let reversed_topology =
        ComplexCore::admit(CandidateInput::unsigned([0_u64, 2, 1], 1, 3, Some(3)).unwrap())
            .unwrap();
    let reversed_realization = EuclideanRealization::admit(
        Arc::clone(&reversed_topology),
        3,
        realization.positions().to_vec(),
        RealizationLimit::DEFAULT,
    )
    .unwrap();
    assert_triangle_differential(
        &reversed_realization,
        [1.0, 7.0, -1.0],
        [4.0, -3.0, 7.0],
        [3.0, -2.0, 0.0],
        [-1.0, -2.0, 3.0],
    );

    assert_differential_rejections(
        &surface,
        &realization,
        scalar.coefficients(),
        field.values(),
    );
    assert_gradient_divergence_matches_stiffness();
}

#[test]
fn frames_are_right_handed_and_store_only_two_axes() {
    let surface = TriangleSurface::admit(tetrahedron(1.0, [0.0; 3])).unwrap();
    let first = surface.first_frame_axes().unwrap();
    let second = surface.second_frame_axes().unwrap();
    let normals = surface.face_unit_normals().unwrap();

    assert!(std::ptr::eq(first, surface.first_frame_axes().unwrap()));
    assert!(std::ptr::eq(second, surface.second_frame_axes().unwrap()));
    assert_eq!(first.len() + second.len(), 6 * surface.face_count());
    for ((first, second), normal) in first
        .chunks_exact(3)
        .zip(second.chunks_exact(3))
        .zip(normals.values().chunks_exact(3))
    {
        let cross = [
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        ];
        for axis in 0..3 {
            assert_close(cross[axis], normal[axis], 8.0e-15);
        }
    }
}

#[test]
fn gradients_and_curvature_obey_translation_and_scale_laws() {
    let base = TriangleSurface::admit(tetrahedron(1.0, [0.0; 3])).unwrap();
    let moved = TriangleSurface::admit(tetrahedron(3.0, [7.0, -5.0, 11.0])).unwrap();
    let base_area = base.surface_area_gradient().unwrap();
    let moved_area = moved.surface_area_gradient().unwrap();
    let base_volume = base.volume_gradient().unwrap();
    let moved_volume = moved.volume_gradient().unwrap();
    for (scaled, value) in moved_area.values().iter().zip(base_area.values()) {
        assert_close(*scaled, 3.0 * value, 2.0e-13);
    }
    for (scaled, value) in moved_volume.values().iter().zip(base_volume.values()) {
        assert_close(*scaled, 9.0 * value, 2.0e-13);
    }

    let rotated_positions = base
        .realization()
        .positions()
        .chunks_exact(3)
        .flat_map(|point| [-point[1] + 4.0, point[0] - 7.0, point[2] + 2.0])
        .collect();
    let rotated_realization = EuclideanRealization::admit(
        Arc::clone(base.realization().topology()),
        3,
        rotated_positions,
        RealizationLimit::DEFAULT,
    )
    .unwrap();
    let rotated = TriangleSurface::admit(rotated_realization).unwrap();
    for (expected, actual) in [
        base.face_unit_normals().unwrap(),
        base.surface_area_gradient().unwrap(),
        base.volume_gradient().unwrap(),
    ]
    .into_iter()
    .zip([
        rotated.face_unit_normals().unwrap(),
        rotated.surface_area_gradient().unwrap(),
        rotated.volume_gradient().unwrap(),
    ]) {
        for (expected, actual) in expected
            .values()
            .chunks_exact(3)
            .zip(actual.values().chunks_exact(3))
        {
            assert_close(actual[0], -expected[1], 2.0e-14);
            assert_close(actual[1], expected[0], 2.0e-14);
            assert_close(actual[2], expected[2], 2.0e-14);
        }
    }

    let curvature = base.gaussian_curvature_measure().unwrap();
    assert_close(curvature.coefficients().iter().sum(), 4.0 * PI, 2.0e-14);

    let disk = TriangleSurface::admit(triangle()).unwrap();
    let boundary_curvature = disk.gaussian_curvature_measure().unwrap();
    assert_close(
        boundary_curvature.coefficients().iter().sum(),
        2.0 * PI,
        2.0e-14,
    );
    assert_eq!(
        disk.volume_gradient().unwrap_err(),
        SurfaceError::BoundaryPresent
    );
}

#[test]
fn connection_retains_only_transport_and_integrability_shares_owner() {
    let surface = TriangleSurface::admit(tetrahedron(1.0, [0.0; 3])).unwrap();
    let levi_civita = surface.levi_civita_connection().unwrap();
    assert_eq!(levi_civita.transports().len(), 2 * surface.edge_count());

    let deviations: Vec<f64> = levi_civita
        .transports()
        .chunks_exact(2)
        .map(|value| -value[1].atan2(value[0]))
        .collect();
    let flat = surface.connection(NonZeroU32::MIN, &deviations).unwrap();
    for value in flat.transports().chunks_exact(2) {
        assert_close(value[0], 1.0, 8.0e-15);
        assert_close(value[1], 0.0, 8.0e-15);
    }

    let cycles = surface
        .realization()
        .topology()
        .integral_dual_cycle_basis()
        .unwrap();
    let holonomy = flat.holonomy(&cycles).unwrap();
    assert!(holonomy.local_error() <= holonomy.limit());
    assert!(holonomy.generator_error() <= holonomy.limit());
    let integrable = flat.require_integrable().unwrap();
    assert!(Arc::ptr_eq(integrable.connection(), &flat));

    let field = integrable.direction_field(0.25).unwrap();
    assert!(Arc::ptr_eq(field.connection(), &flat));
    assert_eq!(field.power_directions().len(), 2 * surface.face_count());
    assert_close(field.crossing_error().unwrap(), 0.0, holonomy.limit());
    let vectors = field.ambient_vector_branch_copy(0).unwrap();
    for vector in vectors.values().chunks_exact(3) {
        assert_close(norm(vector), 1.0, 8.0e-15);
    }
}

#[test]
fn direction_field_power_charges_are_exact_and_quantization_checked() {
    let surface = TriangleSurface::admit(tetrahedron(1.0, [0.0; 3])).unwrap();
    let levi_civita = surface.levi_civita_connection().unwrap();
    for order in [1, 2, 4].map(|value| NonZeroU32::new(value).unwrap()) {
        let deviations = levi_civita
            .transports()
            .chunks_exact(2)
            .map(|value| -f64::from(order.get()) * value[1].atan2(value[0]))
            .collect::<Vec<_>>();
        let integrable = surface
            .connection(order, &deviations)
            .unwrap()
            .require_integrable()
            .unwrap();

        let singularities = integrable
            .direction_field(0.0)
            .unwrap()
            .singularities()
            .unwrap();
        let rotated = integrable
            .direction_field(0.137)
            .unwrap()
            .singularities()
            .unwrap();
        assert_eq!(singularities.symmetry_order(), order);
        assert_eq!(
            singularities.charges().indices(),
            rotated.charges().indices()
        );
        assert_eq!(
            singularities.charges().coefficients(),
            rotated.charges().coefficients()
        );
        assert_eq!(singularities.charges().degree(), 0);
        assert_eq!(
            singularities
                .charges()
                .coefficients()
                .iter()
                .cloned()
                .sum::<BigInt>(),
            BigInt::from(2 * order.get())
        );
        assert!(singularities.maximum_quantization_residual() <= singularities.residual_limit());
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one behavior cluster covers valid orders and adjacent atomic failures"
)]
fn minimum_energy_direction_field_realizes_exact_sphere_power_charges() {
    let realization = octahedron();
    let metric = realization
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    let surface = TriangleSurface::admit(realization).unwrap();
    let homology = IntegralHomology::analyze(
        &surface.realization().topology().chain_complex(),
        [1],
        HomologyLimit::DEFAULT,
    )
    .unwrap();
    let harmonic = metric
        .harmonic_one_form_basis(
            homology.group(1).unwrap(),
            &NativeExecutor::sequential(),
            StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
            WorkLimit::new(u64::MAX),
            &CancellationToken::new(),
        )
        .unwrap();
    let cycles = surface
        .realization()
        .topology()
        .integral_dual_cycle_basis()
        .unwrap();
    let charge_space = surface
        .realization()
        .topology()
        .chain_complex()
        .dual()
        .space(0)
        .unwrap();
    for order in [1, 2, 4].map(|value| NonZeroU32::new(value).unwrap()) {
        let requested = match order.get() {
            1 => charge_space.element([(0, 1.into()), (1, 1.into())]),
            2 => charge_space.element([(0, 1.into()), (1, 1.into()), (2, 1.into()), (3, 1.into())]),
            4 => charge_space.element([
                (0, 2.into()),
                (1, 2.into()),
                (2, 1.into()),
                (3, 1.into()),
                (4, 1.into()),
                (5, 1.into()),
            ]),
            _ => unreachable!(),
        }
        .unwrap();
        let field = surface
            .minimum_energy_direction_field(
                order,
                &metric,
                &harmonic,
                &cycles,
                &requested,
                &[],
                0.25,
                &NativeExecutor::sequential(),
                StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
                WorkLimit::new(u64::MAX),
                &CancellationToken::new(),
            )
            .unwrap_or_else(|error| panic!("order {order} failed: {error:?}"));

        assert_eq!(field.symmetry_order(), order);
        let observed = field.singularities().unwrap();
        assert_eq!(observed.charges().indices(), requested.indices());
        assert_eq!(observed.charges().coefficients(), requested.coefficients());
    }

    let invalid = charge_space.element([(0, BigInt::from(1))]).unwrap();
    assert_eq!(
        surface
            .minimum_energy_direction_field(
                NonZeroU32::new(2).unwrap(),
                &metric,
                &harmonic,
                &cycles,
                &invalid,
                &[],
                0.0,
                &NativeExecutor::sequential(),
                StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
                WorkLimit::new(u64::MAX),
                &CancellationToken::new(),
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::ProblemMismatch)
    );
    let requested = charge_space
        .element([(0, BigInt::from(1)), (1, BigInt::from(1))])
        .unwrap();
    assert_eq!(
        surface
            .minimum_energy_direction_field(
                NonZeroU32::MIN,
                &metric,
                &harmonic,
                &cycles,
                &requested,
                &[],
                0.0,
                &NativeExecutor::sequential(),
                StorageLimit::new(0, 0).unwrap(),
                WorkLimit::new(u64::MAX),
                &CancellationToken::new(),
            )
            .unwrap_err()
            .solve(),
        Some(SolveError::ResourceLimit)
    );
}

#[test]
fn minimum_energy_direction_field_closes_lifted_torus_turns() {
    let torus_realization = torus(8, 7);
    let torus_metric = torus_realization
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    let torus_surface = TriangleSurface::admit(torus_realization).unwrap();
    let torus_homology = IntegralHomology::analyze(
        &torus_surface.realization().topology().chain_complex(),
        [1],
        HomologyLimit::DEFAULT,
    )
    .unwrap();
    let torus_harmonic = torus_metric
        .harmonic_one_form_basis(
            torus_homology.group(1).unwrap(),
            &NativeExecutor::sequential(),
            StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
            WorkLimit::new(u64::MAX),
            &CancellationToken::new(),
        )
        .unwrap();
    let torus_cycles = torus_surface
        .realization()
        .topology()
        .integral_dual_cycle_basis()
        .unwrap();
    let no_charges = torus_surface
        .realization()
        .topology()
        .chain_complex()
        .dual()
        .space(0)
        .unwrap()
        .element([])
        .unwrap();
    let torus_field = torus_surface
        .minimum_energy_direction_field(
            NonZeroU32::new(4).unwrap(),
            &torus_metric,
            &torus_harmonic,
            &torus_cycles,
            &no_charges,
            &[1, 0],
            0.0,
            &NativeExecutor::sequential(),
            StorageLimit::new(u64::MAX, u64::MAX).unwrap(),
            WorkLimit::new(u64::MAX),
            &CancellationToken::new(),
        )
        .unwrap();
    assert_eq!(torus_harmonic.rank(), 2);
    assert_eq!(torus_field.symmetry_order().get(), 4);
    assert!(
        torus_field
            .singularities()
            .unwrap()
            .charges()
            .indices()
            .is_empty()
    );
}

#[test]
fn vertex_normal_and_curvature_algorithms_share_the_field_carrier() {
    let realization = tetrahedron(1.0, [0.0; 3]);
    let metric = realization
        .circumcentric_pairing()
        .unwrap()
        .require_positive()
        .unwrap();
    let surface = TriangleSurface::admit(Arc::clone(&realization)).unwrap();

    for field in [
        surface.uniform_vertex_normals().unwrap(),
        surface.tip_angle_vertex_normals().unwrap(),
        surface.sphere_inscribed_vertex_normals().unwrap(),
    ] {
        assert!(field.is_vertex_supported());
        for vector in field.values().chunks_exact(3) {
            assert_close(norm(vector), 1.0, 2.0e-14);
        }
    }

    let mean = surface.mean_curvature_vectors(&metric).unwrap();
    assert!(mean.is_vertex_supported());
    assert!(mean.values().iter().all(|value| value.is_finite()));
}

#[test]
fn symmetric_connection_retains_power_order_and_projects_explicit_branches() {
    let surface = TriangleSurface::admit(torus(4, 5)).unwrap();
    let order = NonZeroU32::new(4).unwrap();
    let levi_civita = surface.levi_civita_connection().unwrap();
    let deviations = levi_civita
        .transports()
        .chunks_exact(2)
        .map(|value| -f64::from(order.get()) * value[1].atan2(value[0]))
        .collect::<Vec<_>>();
    let connection = surface.connection(order, &deviations).unwrap();
    assert_eq!(connection.symmetry_order(), order);
    for transport in connection.transports().chunks_exact(2) {
        assert_close(transport[0], 1.0, 2.0e-14);
        assert_close(transport[1], 0.0, 2.0e-14);
    }

    let field = connection
        .require_integrable()
        .unwrap()
        .direction_field(0.125)
        .unwrap();
    assert_eq!(field.symmetry_order(), order);
    assert_eq!(field.power_directions().len(), 2 * surface.face_count());
    let singularities = field.singularities().unwrap();
    assert_eq!(singularities.symmetry_order(), order);
    assert_eq!(
        singularities
            .charges()
            .coefficients()
            .iter()
            .cloned()
            .sum::<BigInt>(),
        BigInt::from(0)
    );
    assert!(singularities.maximum_quantization_residual() <= singularities.residual_limit());

    let first = field.ambient_vector_branch_copy(0).unwrap();
    let second = field.ambient_vector_branch_copy(1).unwrap();
    for (first, second) in first
        .values()
        .chunks_exact(3)
        .zip(second.values().chunks_exact(3))
    {
        assert_close(
            first.iter().zip(second).map(|(a, b)| a * b).sum(),
            0.0,
            2.0e-14,
        );
    }
    assert_eq!(
        field.ambient_vector_branch_copy(4).unwrap_err(),
        SurfaceError::IndexOutside
    );
}

#[test]
fn genus_two_cycle_coordinates_certify_identity_transport() {
    let surface = TriangleSurface::admit(torus(4, 5)).unwrap();
    let levi_civita = surface.levi_civita_connection().unwrap();
    let deviations = levi_civita
        .transports()
        .chunks_exact(2)
        .map(|value| -value[1].atan2(value[0]))
        .collect::<Vec<_>>();
    let identity = surface.connection(NonZeroU32::MIN, &deviations).unwrap();
    let cycles = surface
        .realization()
        .topology()
        .integral_dual_cycle_basis()
        .unwrap();
    assert_eq!(cycles.rank(), 2);
    let evidence = identity.holonomy(&cycles).unwrap();
    assert!(evidence.local_error() <= evidence.limit());
    assert!(evidence.generator_error() <= evidence.limit());
    identity.require_integrable().unwrap();
}

#[test]
fn deterministic_nonintegrability_is_stable_and_boundary_connections_are_compact() {
    let surface = TriangleSurface::admit(tetrahedron(1.0, [0.0; 3])).unwrap();
    let mut deviations = vec![0.0; surface.edge_count()];
    deviations[0] = 0.25;
    let connection = surface.connection(NonZeroU32::MIN, &deviations).unwrap();
    assert_eq!(
        connection.require_integrable().unwrap_err(),
        SurfaceError::NotIntegrable
    );
    assert_eq!(
        connection.require_integrable().unwrap_err(),
        SurfaceError::NotIntegrable
    );

    let disk = TriangleSurface::admit(triangle()).unwrap();
    let bounded = disk.levi_civita_connection().unwrap();
    assert!(bounded.interior_edge_indices_copy().is_empty());
    assert!(bounded.transports().is_empty());
    let field = bounded
        .require_integrable()
        .unwrap()
        .direction_field(0.0)
        .unwrap();
    assert_eq!(field.power_directions(), &[1.0, 0.0]);

    let fan = TriangleSurface::admit(nonplanar_disk()).unwrap();
    let boundary = fan.realization().topology().boundary(2).unwrap();
    let expected = (0..fan.edge_count())
        .filter(|&edge| boundary.indptr()[edge + 1] - boundary.indptr()[edge] == 2)
        .collect::<Vec<_>>();
    let bounded = fan.levi_civita_connection().unwrap();
    assert_eq!(&*bounded.interior_edge_indices_copy(), expected);
    assert_eq!(bounded.transports().len(), 2 * expected.len());
    assert_eq!(
        fan.connection(NonZeroU32::MIN, &vec![0.0; fan.edge_count()])
            .unwrap_err(),
        SurfaceError::FieldShape
    );
}

#[test]
fn surface_admission_rejects_non_three_dimensional_realizations() {
    let topology =
        ComplexCore::admit(CandidateInput::unsigned([0_u64, 1, 2], 1, 3, Some(3)).unwrap())
            .unwrap();
    let realization = EuclideanRealization::admit(
        topology,
        2,
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        RealizationLimit::DEFAULT,
    )
    .unwrap();
    assert_eq!(
        TriangleSurface::admit(realization).unwrap_err(),
        SurfaceError::AmbientDimension
    );
}
