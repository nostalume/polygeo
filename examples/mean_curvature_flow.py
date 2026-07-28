import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from polygeo import (
        ORDINARY_FORM,
        AlgorithmError,
        Geometry,
        PositiveHodgeMetric,
        mean_curvature_flow_step,
        plot_cochain,
        plot_geometry,
        prepare_direct,
    )
    from support.meshes import icosphere

    return (
        AlgorithmError,
        Geometry,
        ORDINARY_FORM,
        PositiveHodgeMetric,
        icosphere,
        mean_curvature_flow_step,
        mo,
        np,
        plot_cochain,
        plot_geometry,
        prepare_direct,
    )


@app.cell
def _(mo):
    mo.Html("""
    <style>
    .marimo { max-width: 980px; margin: 0 auto; }
    h1, h2 { letter-spacing: -0.02em; }
    table { display: block; overflow-x: auto; }
    .plotly-graph-div { max-width: 100%; }
    pre, code { overflow-x: auto; }
    </style>
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # One frozen-metric mean-curvature-flow step

    ## Mathematical question

    For a closed triangulated sphere, PolyGeo computes one implicit step
    \[
      (M+\tau K)X_{\rm new}=MX_{\rm old},
    \]
    with $M$ and $K$ frozen from the source geometry. For a smooth radius-$R$ sphere,
    the corresponding frozen implicit contraction is
    \[
      R_{\rm ref}=\frac{R}{1+2\tau/R^2}.
    \]
    How closely does a refined icosphere follow that reference, and which discrete laws
    can be checked independently?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From mathematics to PolyGeo

    | Mathematics | PolyGeo representation |
    |---|---|
    | refined closed sphere | `icosphere(subdivisions, radius)` |
    | optional admitted perturbation | `Geometry.from_positions` then `PositiveHodgeMetric` |
    | frozen $M,H_1>0$ | `PositiveHodgeMetric(source)` |
    | one implicit step | `mean_curvature_flow_step(metric, tau, prepare_direct)` |
    | retained map | `result.output.source`, `.target`, and their exact complex |
    | solver/energy evidence | `result.evidence` |

    The perturbation is used only if the candidate geometry itself admits a represented
    positive Hodge metric; otherwise the study explicitly falls back to the sphere.
    """)
    return


@app.cell
def _(mo):
    refinement = mo.ui.slider(1, 3, value=2, step=1, label="icosphere subdivisions")
    time_step = mo.ui.slider(0.01, 0.20, value=0.05, step=0.01, label="time step")
    perturb = mo.ui.checkbox(
        value=False, label="request deterministic 3% radial perturbation"
    )
    mo.vstack([refinement, time_step, perturb])
    return perturb, refinement, time_step


@app.cell
def _(
    AlgorithmError,
    Geometry,
    PositiveHodgeMetric,
    icosphere,
    np,
    perturb,
    refinement,
):
    sphere_radius = 1.0
    flow_domain, sphere_geometry = icosphere(
        refinement.value, radius=float(sphere_radius)
    )
    sphere_positions = sphere_geometry.positions
    unit_directions = (
        sphere_positions / np.linalg.norm(sphere_positions, axis=1)[:, None]
    )
    radial_pattern = (
        unit_directions[:, 0] * unit_directions[:, 1] + 0.5 * unit_directions[:, 2]
    )
    candidate_positions = sphere_positions * (1.0 + 0.03 * radial_pattern)[:, None]
    perturbation_requested = bool(perturb.value)
    perturbation_admitted = False
    if perturbation_requested:
        candidate_geometry = Geometry.from_positions(flow_domain, candidate_positions)
        try:
            source_metric = PositiveHodgeMetric(candidate_geometry)
            source_geometry = candidate_geometry
            perturbation_admitted = True
        except AlgorithmError:
            source_geometry = sphere_geometry
            source_metric = PositiveHodgeMetric(source_geometry)
    else:
        source_geometry = sphere_geometry
        source_metric = PositiveHodgeMetric(source_geometry)
    return (
        flow_domain,
        perturbation_admitted,
        perturbation_requested,
        source_geometry,
        source_metric,
        sphere_radius,
    )


@app.cell
def _(
    Geometry,
    ORDINARY_FORM,
    PositiveHodgeMetric,
    mean_curvature_flow_step,
    np,
    plot_cochain,
    plot_geometry,
    prepare_direct,
    source_geometry,
    source_metric,
    sphere_radius,
    time_step,
):
    from scipy.sparse import diags

    retained_source = source_geometry.positions
    flow_result = mean_curvature_flow_step(
        source_metric, float(time_step.value), prepare_direct
    )
    target_geometry = flow_result.output.target
    source_positions = source_geometry.positions
    target_positions = target_geometry.positions
    displacement = np.linalg.norm(target_positions - source_positions, axis=1)

    mass = diags(source_metric.weights(0), format="csr")
    derivative = source_geometry.complex.boundary_matrix(1).T
    stiffness = derivative.T @ diags(source_metric.weights(1)) @ derivative
    equation_terms = (
        mass + time_step.value * stiffness
    ) @ target_positions - mass @ source_positions
    equation_residual = float(np.max(np.abs(equation_terms), initial=0.0))
    source_edges = derivative @ source_positions
    target_edges = derivative @ target_positions
    edge_weights = source_metric.weights(1)[:, None]
    frozen_energy_before = 0.5 * float(
        np.sum(edge_weights * source_edges * source_edges)
    )
    frozen_energy_after = 0.5 * float(
        np.sum(edge_weights * target_edges * target_edges)
    )
    energy_evidence_error = max(
        abs(frozen_energy_before - flow_result.evidence.energy_before),
        abs(frozen_energy_after - flow_result.evidence.energy_after),
    )
    vertex_weights = source_metric.weights(0)
    centroid_before = vertex_weights @ source_positions / np.sum(vertex_weights)
    centroid_after = vertex_weights @ target_positions / np.sum(vertex_weights)
    centroid_residual = float(
        np.max(np.abs(centroid_after - centroid_before), initial=0.0)
    )
    identity_laws = (
        flow_result.output.source is source_geometry
        and target_geometry.complex is source_geometry.complex
        and flow_result.output.target_dimension == source_geometry.ambient_dimension
    )
    source_mutation_error = float(
        np.max(np.abs(source_geometry.positions - retained_source), initial=0.0)
    )

    target_radii = np.linalg.norm(target_positions - centroid_after, axis=1)
    reference_radius = sphere_radius / (1.0 + 2.0 * time_step.value / sphere_radius**2)
    mean_target_radius = float(
        np.sum(vertex_weights * target_radii) / np.sum(vertex_weights)
    )
    contraction_error = abs(mean_target_radius - reference_radius)
    radial_spread = float(np.max(target_radii) - np.min(target_radii))

    scale_factor = 2.0
    scaled_source = Geometry.from_positions(
        source_geometry.complex, scale_factor * source_positions
    )
    scaled_result = mean_curvature_flow_step(
        PositiveHodgeMetric(scaled_source),
        float(time_step.value * scale_factor**2),
        prepare_direct,
    )
    scale_covariance_error = float(
        np.max(
            np.abs(
                scaled_result.output.target.positions / scale_factor - target_positions
            ),
            initial=0.0,
        )
    )
    source_figure = plot_geometry(source_geometry, title="Source geometry")
    target_figure = plot_geometry(target_geometry, title="Target after one frozen step")
    displacement_form = source_geometry.complex.cochain_space(0).form(
        displacement, ORDINARY_FORM
    )
    displacement_figure = plot_cochain(
        source_geometry,
        displacement_form,
        title="One-step displacement magnitude",
    )
    return (
        centroid_residual,
        contraction_error,
        displacement_figure,
        energy_evidence_error,
        equation_residual,
        flow_result,
        frozen_energy_after,
        frozen_energy_before,
        identity_laws,
        mean_target_radius,
        radial_spread,
        reference_radius,
        scale_covariance_error,
        source_figure,
        source_mutation_error,
        target_figure,
    )


@app.cell
def _(
    flow_domain,
    mo,
    perturbation_admitted,
    perturbation_requested,
    refinement,
    time_step,
):
    perturbation_status = (
        "used"
        if perturbation_admitted
        else "requested but not positive-Hodge-admitted; sphere used"
        if perturbation_requested
        else "not requested"
    )
    mo.md(rf"""
    ## Computation

    The mesh has subdivision level **{refinement.value}**,
    **{flow_domain.simplex_count(0)} vertices**, and
    **{flow_domain.simplex_count(2)} faces**. The represented time step is
    **{time_step.value:.2f}**. Perturbation status: **{perturbation_status}**.
    The source metric is frozen for all three coordinate solves.

    ## Visualization

    Source, target, and displacement magnitude are displayed vertically.
    """)
    return


@app.cell
def _(source_figure):
    source_figure
    return


@app.cell
def _(target_figure):
    target_figure
    return


@app.cell
def _(displacement_figure):
    displacement_figure
    return


@app.cell
def _(
    centroid_residual,
    contraction_error,
    energy_evidence_error,
    equation_residual,
    flow_result,
    frozen_energy_after,
    frozen_energy_before,
    identity_laws,
    mean_target_radius,
    mo,
    radial_spread,
    reference_radius,
    scale_covariance_error,
    source_mutation_error,
):
    evidence = flow_result.evidence
    solve_residual = max(solve.residual_norm for solve in evidence.solves)
    mo.md(rf"""
    ## Evaluation

    | Independent law | Observed |
    |---|---:|
    | coordinate equation residual max | `{equation_residual:.3e}` |
    | largest certified solve residual | `{solve_residual:.3e}` |
    | independently recomputed frozen energy before | `{frozen_energy_before:.9g}` |
    | independently recomputed frozen energy after | `{frozen_energy_after:.9g}` |
    | energy / retained-evidence difference | `{energy_evidence_error:.3e}` |
    | old-mass weighted centroid residual max | `{centroid_residual:.3e}` |
    | source mutation error | `{source_mutation_error:.3e}` |
    | source/target exact identity laws | `{identity_laws}` |
    | scale covariance error, $s=2,\ \tau\mapsto s^2\tau$ | `{scale_covariance_error:.3e}` |
    | smooth-sphere reference radius | `{reference_radius:.9g}` |
    | weighted mean target radius | `{mean_target_radius:.9g}` |
    | contraction forward error | `{contraction_error:.3e}` |
    | target radial spread | `{radial_spread:.3e}` |

    Energy, centroid, equation, identity, and scale laws are recomputed through public
    geometry, metric weights, canonical boundary matrices, and retained output objects.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    This is evidence for exactly one frozen-metric implicit step. The smooth sphere formula
    is a forward reference, not an exact law for a finite triangulation; with the optional
    perturbation it is only a scale for comparison. The study does **not** claim a
    multi-step integrator, convergence to continuous mean-curvature flow, preservation of
    embeddedness or mesh quality, avoidance of singularities, or long-time energy behavior.
    Re-admitting a target metric and choosing a subsequent time step are separate tasks.
    """)
    return


if __name__ == "__main__":
    app.run()
