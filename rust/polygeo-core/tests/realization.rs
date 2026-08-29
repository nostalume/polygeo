use std::sync::Arc;

use polygeo_core::{
    CandidateInput, CircumcentricPairing, ComplexCore, EuclideanRealization, MetricError,
    NondegenerateCapability, NondegeneratePairing, PairingCapability, PositiveMetric,
    RealizationError, RealizationLimit, StorageLimit, WorkLimit,
};

fn triangle() -> Arc<ComplexCore> {
    ComplexCore::admit(CandidateInput::signed([0_i64, 1, 2], 1, 3, None).unwrap()).unwrap()
}

fn right_triangle() -> [f64; 6] {
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
}

fn realized_triangle(positions: [f64; 6]) -> Arc<EuclideanRealization> {
    EuclideanRealization::admit(triangle(), 2, positions.to_vec(), RealizationLimit::DEFAULT)
        .unwrap()
}

fn simplex(dimension: usize, scale: f64) -> Arc<EuclideanRealization> {
    let vertices = (0..=dimension)
        .map(|vertex| i64::try_from(vertex).unwrap())
        .collect::<Vec<_>>();
    let topology = ComplexCore::admit(
        CandidateInput::signed(vertices, 1, dimension + 1, Some(dimension + 1)).unwrap(),
    )
    .unwrap();
    let mut positions = vec![0.0; (dimension + 1) * dimension];
    for axis in 0..dimension {
        positions[(axis + 1) * dimension + axis] = scale;
    }
    EuclideanRealization::admit(topology, dimension, positions, RealizationLimit::DEFAULT).unwrap()
}

#[test]
fn realization_owns_one_position_copy_and_retains_topology() {
    let owner = triangle();
    let mut positions = right_triangle();
    let position_buffer = positions.to_vec();
    let transferred = position_buffer.as_ptr();
    let realization = EuclideanRealization::admit(
        Arc::clone(&owner),
        2,
        position_buffer,
        RealizationLimit::DEFAULT,
    )
    .unwrap();

    positions.fill(9.0);
    assert!(Arc::ptr_eq(realization.topology(), &owner));
    assert_eq!(realization.ambient_dimension(), 2);
    assert_eq!(realization.positions(), &right_triangle());
    assert_eq!(realization.positions().as_ptr(), transferred);
}

#[test]
fn realization_computes_primal_and_cached_dual_rows() {
    let realization = EuclideanRealization::admit(
        triangle(),
        2,
        right_triangle().to_vec(),
        RealizationLimit::DEFAULT,
    )
    .unwrap();

    assert_eq!(realization.primal_measures(0).unwrap(), &[1.0; 3]);
    assert_eq!(realization.primal_measures(2).unwrap(), &[0.5]);
    assert_eq!(realization.dual_measures(2).unwrap(), &[1.0]);
    assert_eq!(realization.dual_measures(1).unwrap(), &[0.5, 0.5, 0.0]);
    assert_eq!(realization.dual_measures(0).unwrap(), &[0.25, 0.125, 0.125]);
    assert!(std::ptr::eq(
        realization.dual_measures(0).unwrap(),
        realization.dual_measures(0).unwrap()
    ));
}

#[test]
fn realization_rejects_before_publication() {
    let owner = triangle();
    assert_eq!(
        EuclideanRealization::admit(
            Arc::clone(&owner),
            1,
            vec![0.0; 3],
            RealizationLimit::DEFAULT,
        )
        .unwrap_err(),
        RealizationError::AmbientDimension
    );
    assert_eq!(
        EuclideanRealization::admit(
            Arc::clone(&owner),
            2,
            vec![0.0; 5],
            RealizationLimit::DEFAULT,
        )
        .unwrap_err(),
        RealizationError::PositionShape
    );
    assert_eq!(
        EuclideanRealization::admit(
            Arc::clone(&owner),
            2,
            vec![0.0, 0.0, f64::NAN, 0.0, 0.0, 1.0],
            RealizationLimit::DEFAULT,
        )
        .unwrap_err(),
        RealizationError::NonFinite
    );
    assert_eq!(
        EuclideanRealization::admit(
            owner,
            2,
            right_triangle().to_vec(),
            RealizationLimit::new(
                StorageLimit::new(
                    414,
                    RealizationLimit::DEFAULT
                        .storage()
                        .peak_live_logical_bytes()
                )
                .unwrap(),
                RealizationLimit::DEFAULT.coefficient_bits(),
                RealizationLimit::DEFAULT.exact_steps(),
            ),
        )
        .unwrap_err(),
        RealizationError::RetainedLogicalBytes {
            required: 415,
            limit: 414,
        }
    );
    EuclideanRealization::admit(
        triangle(),
        2,
        right_triangle().to_vec(),
        RealizationLimit::new(
            StorageLimit::new(
                415,
                RealizationLimit::DEFAULT
                    .storage()
                    .peak_live_logical_bytes(),
            )
            .unwrap(),
            RealizationLimit::DEFAULT.coefficient_bits(),
            RealizationLimit::DEFAULT.exact_steps(),
        ),
    )
    .unwrap();
}

#[test]
fn realization_limits_report_storage_and_exact_axes() {
    let storage = StorageLimit::new(0, 0).unwrap();
    let limit = RealizationLimit::new(storage, 4096, WorkLimit::new(100_000));
    let error =
        EuclideanRealization::admit(triangle(), 2, right_triangle().to_vec(), limit).unwrap_err();
    assert_eq!(error.reason(), "resource_limit");
    assert!(matches!(
        error.resource_limit(),
        Some(("retained_logical_bytes", required, 0)) if required > 0
    ));
}

#[test]
fn realization_preflights_peak_and_coefficient_growth() {
    let peak = RealizationLimit::new(
        StorageLimit::new(415, 415).unwrap(),
        RealizationLimit::DEFAULT.coefficient_bits(),
        RealizationLimit::DEFAULT.exact_steps(),
    );
    assert!(matches!(
        EuclideanRealization::admit(triangle(), 2, right_triangle().to_vec(), peak)
            .unwrap_err()
            .resource_limit(),
        Some(("peak_live_logical_bytes", required, 415)) if required > 415
    ));

    let narrow = RealizationLimit::new(
        RealizationLimit::DEFAULT.storage(),
        10,
        RealizationLimit::DEFAULT.exact_steps(),
    );
    assert!(matches!(
        EuclideanRealization::admit(
            triangle(),
            2,
            vec![0.0, 0.0, 1.0, 0.0, 1.0, f64::MIN_POSITIVE],
            narrow,
        )
        .unwrap_err()
        .resource_limit(),
        Some(("coefficient_bits", required, 10)) if required > 10
    ));
}

#[test]
fn exact_steps_are_cumulative_across_primal_fallbacks() {
    let topology =
        ComplexCore::admit(CandidateInput::signed([0_i64, 1, 2, 3, 4, 5], 2, 3, Some(6)).unwrap())
            .unwrap();
    let height = f64::MIN_POSITIVE;
    let positions = vec![
        0.0, 0.0, 1.0, 0.0, 1.0, height, 2.0, 0.0, 3.0, 0.0, 3.0, height,
    ];
    let limit = RealizationLimit::new(
        RealizationLimit::DEFAULT.storage(),
        RealizationLimit::DEFAULT.coefficient_bits(),
        WorkLimit::new(60),
    );
    EuclideanRealization::admit(triangle(), 2, vec![0.0, 0.0, 1.0, 0.0, 1.0, height], limit)
        .unwrap();
    assert!(matches!(
        EuclideanRealization::admit(topology, 2, positions, limit)
            .unwrap_err()
            .resource_limit(),
        Some(("exact_steps", required, 60)) if required > 60
    ));
}

#[test]
fn simplex_measures_preserve_dimension_and_extreme_binary64_scale() {
    let mut factorial = 1.0;
    for dimension in 0..=5 {
        if dimension > 1 {
            factorial *= f64::from(u32::try_from(dimension).unwrap());
        }
        let realization = simplex(dimension, 1.0);
        assert_eq!(
            realization.primal_measures(dimension).unwrap(),
            &[1.0 / factorial]
        );
    }
    for scale in [1.0e-150, 1.0e150] {
        let realization = simplex(2, scale);
        assert_eq!(
            realization.primal_measures(2).unwrap(),
            &[scale * scale / 2.0]
        );
    }
}

#[test]
fn exact_fallback_preserves_representable_subnormal_measure() {
    let topology = triangle();
    let height = f64::MIN_POSITIVE;
    let realization = EuclideanRealization::admit(
        topology,
        2,
        vec![0.0, 0.0, 1.0, 0.0, 1.0, height],
        RealizationLimit::DEFAULT,
    )
    .unwrap();

    assert_eq!(realization.primal_measures(2).unwrap(), &[height / 2.0]);
}

fn pairing_owner(value: &impl PairingCapability) -> &Arc<EuclideanRealization> {
    value.realization()
}

fn nondegenerate_owner(value: &impl NondegenerateCapability) -> &Arc<EuclideanRealization> {
    value.realization()
}

#[test]
fn metric_refinement_is_cached_evidence_over_one_realization() {
    let height = 3.0_f64.sqrt() / 2.0;
    let realization = realized_triangle([0.0, 0.0, 1.0, 0.0, 0.5, height]);
    let pairing = realization.circumcentric_pairing().unwrap();
    let coefficients = pairing.hodge_coefficients_slice(1).unwrap();
    let pointer = coefficients.as_ptr();
    let nondegenerate = NondegeneratePairing::try_from(pairing.clone()).unwrap();
    let positive = PositiveMetric::try_from(nondegenerate.clone()).unwrap();
    let forgotten: CircumcentricPairing = positive.clone().into();

    assert!(Arc::ptr_eq(pairing_owner(&forgotten), &realization));
    assert!(Arc::ptr_eq(nondegenerate_owner(&positive), &realization));
    assert_eq!(
        pointer,
        forgotten.hodge_coefficients_slice(1).unwrap().as_ptr()
    );
    assert!(coefficients.iter().all(|value| *value > 0.0));
}

#[test]
fn metric_refinement_reports_zero_before_indefiniteness() {
    let right = realized_triangle(right_triangle())
        .circumcentric_pairing()
        .unwrap();
    assert_eq!(
        NondegeneratePairing::try_from(right).unwrap_err(),
        MetricError::Degenerate {
            degree: 1,
            index: 2,
        }
    );

    let obtuse = realized_triangle([0.0, 0.0, 2.0, 0.0, 0.5, 0.1])
        .circumcentric_pairing()
        .unwrap();
    let nondegenerate = NondegeneratePairing::try_from(obtuse).unwrap();
    assert_eq!(
        PositiveMetric::try_from(nondegenerate).unwrap_err(),
        MetricError::Indefinite
    );
}

#[test]
fn concurrent_pairing_access_publishes_one_coefficient_buffer() {
    let realization = realized_triangle([0.0, 0.0, 1.0, 0.0, 0.5, 3.0_f64.sqrt() / 2.0]);
    let addresses = (0..8)
        .map(|_| {
            let realization = Arc::clone(&realization);
            std::thread::spawn(move || {
                realization
                    .circumcentric_pairing()
                    .unwrap()
                    .hodge_coefficients_slice(1)
                    .unwrap()
                    .as_ptr() as usize
            })
        })
        .map(|thread| thread.join().unwrap())
        .collect::<Vec<_>>();
    assert!(addresses.iter().all(|address| *address == addresses[0]));
}

#[test]
fn exact_fallback_obeys_the_cumulative_step_limit() {
    let budget = RealizationLimit::new(
        RealizationLimit::DEFAULT.storage(),
        RealizationLimit::DEFAULT.coefficient_bits(),
        WorkLimit::new(0),
    );
    let result = EuclideanRealization::admit(
        triangle(),
        2,
        vec![0.0, 0.0, 1.0, 0.0, 1.0, f64::MIN_POSITIVE],
        budget,
    );
    assert!(matches!(
        result.unwrap_err().resource_limit(),
        Some(("exact_steps", required, 0)) if required > 0
    ));
}

#[test]
fn failed_lazy_metric_publication_can_be_retried() {
    let realization = EuclideanRealization::admit(
        triangle(),
        2,
        right_triangle().to_vec(),
        RealizationLimit::new(
            RealizationLimit::DEFAULT.storage(),
            RealizationLimit::DEFAULT.coefficient_bits(),
            WorkLimit::new(0),
        ),
    )
    .unwrap();
    assert_eq!(
        realization.circumcentric_pairing().unwrap_err(),
        RealizationError::ExactSteps {
            required: 1,
            limit: 0
        }
    );
    assert_eq!(
        realization.circumcentric_pairing().unwrap_err(),
        RealizationError::ExactSteps {
            required: 1,
            limit: 0
        }
    );
}

#[test]
fn unrepresentable_pairing_does_not_hide_representable_dual_rows() {
    let realization = realized_triangle([0.0, 0.0, 1.0e170, 0.0, 0.0, 1.0e-170]);
    assert_eq!(
        realization.dual_measures(1).unwrap()[0].to_bits(),
        5.0e-171_f64.to_bits()
    );
    assert_eq!(
        realization.circumcentric_pairing().unwrap_err(),
        RealizationError::Unrepresentable
    );
    assert!(std::ptr::eq(
        realization.dual_measures(1).unwrap(),
        realization.dual_measures(1).unwrap()
    ));
}
