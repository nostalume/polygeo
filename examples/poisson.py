import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import math

    import marimo as mo
    import numpy as np

    from polygeo import (
        ORDINARY_FORM,
        PositiveHodgeMetric,
        assemble_poisson,
        impose_mean_zero,
        plot_cochain,
        prepare_direct,
    )
    from support.meshes import icosphere

    return (
        ORDINARY_FORM,
        PositiveHodgeMetric,
        assemble_poisson,
        icosphere,
        impose_mean_zero,
        math,
        mo,
        np,
        plot_cochain,
        prepare_direct,
    )


@app.cell
def _(mo):
    mo.Html(
        """
        <style>
        .marimo { max-width: 980px; margin: 0 auto; }
        h1, h2 { letter-spacing: -0.02em; }
        table { display: block; overflow-x: auto; }
        .plotly-graph-div { max-width: 100%; }
        pre, code { overflow-x: auto; }
        </style>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Positive-Hodge Poisson problem on an icosphere

    ## Mathematical question

    On a sphere of radius \(R\), the degree-two zonal harmonic

    \[
    u=\frac12\left(3(z/R)^2-1\right)
    \]

    satisfies the positive-Laplacian equation

    \[
    \Delta u=f,\qquad f=\frac{6u}{R^2}.
    \]

    For the discrete closed surface, \(M=\operatorname{diag}(H_0)\) and
    \(K=M\Delta\), so the strong pointwise equation is equivalent to
    \(Ku=Mf\). Compatibility admits a solution; the weighted mean-zero gauge
    selects one representative from the constant nullspace.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From mathematics to PolyGeo

    | Mathematics | PolyGeo public API |
    |---|---|
    | positive circumcentric Hodge metric | `PositiveHodgeMetric(geometry)` |
    | pointwise source \(f\) | canonical degree-zero ordinary form |
    | \(\Delta u=f\) | `assemble_poisson(metric, source)` |
    | compatibility and weighted gauge | `impose_mean_zero(metric, source)` |
    | one prepared anchored solve | `problem.solve(prepare_direct)` |
    | source, solution, signed error | `plot_cochain(...)` |

    Compatibility, gauge, physical residual `Ku-Mf`, and forward error are independent
    claims. A small linear-system residual alone does not prove that the computed
    field is close to the analytic harmonic.
    """)
    return


@app.cell
def _(mo):
    subdivisions = mo.ui.slider(
        0, 3, value=2, step=1, label="bounded icosphere subdivisions"
    )
    subdivisions
    return (subdivisions,)


@app.cell
def _(mo):
    sphere_radius = mo.ui.slider(0.5, 2.0, value=1.0, step=0.25, label="sphere radius")
    sphere_radius
    return (sphere_radius,)


@app.cell
def _(np):
    def weighted_mean(values, weights):
        return float(np.dot(weights, values) / np.sum(weights))

    def weighted_rms(values, weights):
        return float(np.sqrt(np.sum(weights * values * values) / np.sum(weights)))

    return weighted_rms, weighted_mean


@app.cell
def _(
    ORDINARY_FORM,
    PositiveHodgeMetric,
    assemble_poisson,
    icosphere,
    impose_mean_zero,
    np,
    prepare_direct,
    sphere_radius,
    subdivisions,
    weighted_rms,
    weighted_mean,
):
    radius = float(sphere_radius.value)
    domain, geometry = icosphere(subdivisions.value, radius)
    metric = PositiveHodgeMetric(geometry)
    weights = metric.weights(0)
    vertex_space = domain.cochain_space(0)

    analytic_raw = 0.5 * (3.0 * (geometry.positions[:, 2] / radius) ** 2 - 1.0)
    analytic_values = analytic_raw - weighted_mean(analytic_raw, weights)
    source_values = 6.0 * analytic_raw / radius**2
    source = vertex_space.form(source_values, ORDINARY_FORM)

    assembled = assemble_poisson(metric, source)
    problem = impose_mean_zero(metric, source)
    solution = problem.solve(prepare_direct)
    solution_values = solution.form.coefficients()
    error_values = solution_values - analytic_values
    error = vertex_space.form(error_values, ORDINARY_FORM)

    compatibility_residual = abs(float(np.dot(weights, source_values)))
    gauge_residual = abs(float(np.dot(weights, solution_values)))
    physical_residual_values = (
        assembled.operator.apply(solution.form).coefficients() - source_values
    )
    physical_residual = weighted_rms(physical_residual_values, weights)
    forward_rms_error = weighted_rms(error_values, weights)
    return (
        analytic_values,
        compatibility_residual,
        domain,
        error,
        forward_rms_error,
        gauge_residual,
        geometry,
        metric,
        physical_residual,
        problem,
        radius,
        solution,
        source,
        source_values,
        weights,
    )


@app.cell
def _(error, geometry, plot_cochain, solution, source):
    source_figure = plot_cochain(
        geometry, source, title="Analytic pointwise source 6u / R^2"
    )
    solution_figure = plot_cochain(
        geometry, solution.form, title="Mean-zero discrete Poisson solution"
    )
    error_figure = plot_cochain(
        geometry, error, title="Signed forward error: discrete solution - analytic u"
    )
    return error_figure, solution_figure, source_figure


@app.cell
def _(
    ORDINARY_FORM,
    PositiveHodgeMetric,
    assemble_poisson,
    icosphere,
    impose_mean_zero,
    math,
    np,
    prepare_direct,
    radius,
    weighted_rms,
    weighted_mean,
):
    convergence_rows = []
    previous_error = None
    for study_level in range(4):
        study_domain, study_geometry = icosphere(study_level, radius)
        study_metric = PositiveHodgeMetric(study_geometry)
        study_weights = study_metric.weights(0)
        study_space = study_domain.cochain_space(0)
        study_raw = 0.5 * (3.0 * (study_geometry.positions[:, 2] / radius) ** 2 - 1.0)
        study_exact = study_raw - weighted_mean(study_raw, study_weights)
        study_source_values = 6.0 * study_raw / radius**2
        study_source = study_space.form(study_source_values, ORDINARY_FORM)
        study_assembled = assemble_poisson(study_metric, study_source)
        study_problem = impose_mean_zero(study_metric, study_source)
        study_solution = study_problem.solve(prepare_direct)
        study_values = study_solution.form.coefficients()
        study_error = weighted_rms(study_values - study_exact, study_weights)
        study_physical = weighted_rms(
            study_assembled.operator.apply(study_solution.form).coefficients()
            - study_source_values,
            study_weights,
        )
        observed_rate = (
            float("nan")
            if previous_error is None
            else math.log(previous_error / study_error, 2.0)
        )
        convergence_rows.append(
            (
                study_level,
                study_domain.simplex_count(0),
                study_error,
                study_physical,
                observed_rate,
            )
        )
        previous_error = study_error
    return (convergence_rows,)


@app.cell
def _(mo, radius, subdivisions):
    mo.md(
        rf"""
    ## Computation

    The active mesh is a positive-Hodge icosphere with subdivision
    **{subdivisions.value}** and radius **{radius:.2f}**. The analytic harmonic is
    sampled at vertices, its prescribed source is assembled with
    `assemble_poisson`, compatibility and the weighted gauge are admitted by
    `impose_mean_zero`, and the anchored system is factored by `prepare_direct`.

    ## Visualization

    The three cochains are deliberately separate and vertically ordered: source,
    numerical solution, then signed forward error.
    """
    )
    return


@app.cell
def _(source_figure):
    source_figure
    return


@app.cell
def _(solution_figure):
    solution_figure
    return


@app.cell
def _(error_figure):
    error_figure
    return


@app.cell
def _(
    compatibility_residual,
    convergence_rows,
    forward_rms_error,
    gauge_residual,
    mo,
    physical_residual,
    problem,
    solution,
):
    evidence = problem.compatibility_evidence
    convergence_table = "\n".join(
        (
            f"| {level} | {vertices} | {error:.6e} | {residual:.3e} | "
            + ("—" if level == 0 else f"{rate:.3f}")
            + " |"
        )
        for level, vertices, error, residual, rate in convergence_rows
    )
    mo.md(
        rf"""
    ## Evaluation

    | Independent quantity | Observed |
    |---|---:|
    | compatibility \(\left|\\mathbf{{1}}^T M f\\right|\) | `{compatibility_residual:.3e}` |
    | compatibility admission limit | `{evidence.limit * evidence.scale:.3e}` |
    | weighted gauge \(\left|\\mathbf{{1}}^T M u_h\\right|\) | `{gauge_residual:.3e}` |
    | physical residual weighted RMS | `{physical_residual:.3e}` |
    | solver-reported relative residual | `{solution.relative_residual:.3e}` |
    | forward error weighted RMS | `{forward_rms_error:.6e}` |

    | Subdivision | Vertices | Forward RMS error | Physical RMS residual | Error ratio rate |
    |---:|---:|---:|---:|---:|
    {convergence_table}

    The physical residual is recomputed by applying the public assembled operator.
    The forward RMS column instead compares against the known zonal harmonic; it is
    the convergence quantity and must not be replaced by the backward residual.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    The source is compatible because the degree-two spherical harmonic has zero
    weighted mean on these symmetric icospheres. Compatibility removes the constant
    obstruction; the weighted gauge removes the solution ambiguity. The strong
    physical residual checks the solved discrete equation, while forward RMS error
    measures approximation of the continuum field.

    Refinement can reduce forward error even when the direct-solve residual is already
    near floating-point precision. Radius changes the source by the expected
    inverse-square law without changing the dimensionless harmonic.
    """)
    return


if __name__ == "__main__":
    app.run()
