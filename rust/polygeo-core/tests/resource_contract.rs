use polygeo_core::{
    chain::BigIntEncoding, chain::Csr, chain::CsrBuildLimit, chain::IntegerRing,
    chain::RationalField, chain::ReducedFractionEncoding, solve::StorageLimit, solve::WorkLimit,
    topology::CandidateInput, topology::Complex as ComplexCore,
};

fn boundary() -> polygeo_core::chain::LinearMap<
    polygeo_core::chain::IntegerRing,
    polygeo_core::chain::Chain,
    polygeo_core::chain::Chain,
> {
    let candidate = CandidateInput::signed([0_i64, 1, 2], 1, 3, Some(3)).unwrap();
    ComplexCore::admit(candidate)
        .unwrap()
        .chain_complex()
        .boundary(2)
        .unwrap()
}

#[test]
fn rational_csr_accounts_for_both_stored_integer_components() {
    let map = boundary();
    let integer = Csr::estimate(&map, BigIntEncoding).unwrap();
    let rational = Csr::estimate(
        &map.over(RationalField::new(IntegerRing)),
        ReducedFractionEncoding,
    )
    .unwrap();
    assert_eq!(
        rational.coefficient_bits_bound(),
        integer.coefficient_bits_bound()
    );
    assert!(rational.retained_logical_bytes_bound() > integer.retained_logical_bytes_bound());
}

#[test]
fn storage_limit_enforces_its_lifecycle_relation() {
    assert!(StorageLimit::new(7, 6).is_none());
    let limit = StorageLimit::new(6, 7).unwrap();
    assert_eq!(limit.retained_logical_bytes(), 6);
    assert_eq!(limit.peak_live_logical_bytes(), 7);
    assert_eq!(WorkLimit::new(11).steps(), 11);
}

#[test]
fn csr_matching_limit_succeeds_and_each_semantic_ceiling_rejects() {
    let map = boundary();
    let estimate = Csr::estimate(&map, BigIntEncoding).unwrap();
    assert!(estimate.peak_live_logical_bytes_bound() > estimate.retained_logical_bytes_bound());
    assert!(
        u64::try_from(estimate.scratch_entries_bound()).unwrap() <= estimate.scalar_steps_bound()
    );

    Csr::build(&map, BigIntEncoding, CsrBuildLimit::for_estimate(estimate)).unwrap();

    let retained = estimate.retained_logical_bytes_bound();
    let peak = estimate.peak_live_logical_bytes_bound();
    let cases = [
        (
            CsrBuildLimit::for_estimate(estimate)
                .with_storage(StorageLimit::new(retained - 1, peak).unwrap()),
            ("retained_logical_bytes", retained, retained - 1),
        ),
        (
            CsrBuildLimit::for_estimate(estimate)
                .with_storage(StorageLimit::new(retained, peak - 1).unwrap()),
            ("peak_live_logical_bytes", peak, peak - 1),
        ),
        (
            CsrBuildLimit::for_estimate(estimate)
                .with_coefficient_bits(estimate.coefficient_bits_bound() - 1),
            (
                "coefficient_bits",
                estimate.coefficient_bits_bound(),
                estimate.coefficient_bits_bound() - 1,
            ),
        ),
        (
            CsrBuildLimit::for_estimate(estimate)
                .with_scalar_steps(WorkLimit::new(estimate.scalar_steps_bound() - 1)),
            (
                "scalar_steps",
                estimate.scalar_steps_bound(),
                estimate.scalar_steps_bound() - 1,
            ),
        ),
    ];
    for (limit, detail) in cases {
        let error = Csr::build(&map, BigIntEncoding, limit).unwrap_err();
        assert_eq!(error.resource_limit(), Some(detail));
    }
}
