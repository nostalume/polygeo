import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from polygeo import (
        ORDINARY_FORM,
        CochainSubspace,
        Complex,
        Geometry,
        PositiveHodgeMetric,
        harmonic_extension,
        plot_cochain,
        prepare_direct,
        topological_boundary,
    )
    from support.meshes import annulus

    return (
        CochainSubspace,
        Complex,
        Geometry,
        ORDINARY_FORM,
        PositiveHodgeMetric,
        annulus,
        harmonic_extension,
        mo,
        np,
        plot_cochain,
        prepare_direct,
        topological_boundary,
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
    # Harmonic extension on an annulus

    ## Mathematical question

    On the annulus $r_{\rm in}\leq r\leq r_{\rm out}$, prescribe constants
    $u_{\rm in}$ and $u_{\rm out}$ on its two boundary components. We ask whether
    PolyGeo's discrete Dirichlet solution
    \[
      (Ku)_I=0,\qquad u_B=g
    \]
    approaches the rotationally symmetric continuum solution
    \[
      u_\star(r)=u_{\rm in}
      +(u_{\rm out}-u_{\rm in})
      \frac{\log(r/r_{\rm in})}{\log(r_{\rm out}/r_{\rm in})}.
    \]

    The study separates exact algebraic laws (boundary restoration and the interior
    equation) from forward error against $u_\star$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From mathematics to PolyGeo

    | Mathematics | PolyGeo representation |
    |---|---|
    | deterministic Delaunay annulus | `annulus(rings, 4 * rings)` |
    | represented $H_k>0$ | `PositiveHodgeMetric(geometry)` |
    | canonical vertex boundary $B$ | `topological_boundary(domain).mask(0)` |
    | exact parent-bound boundary data | `CochainSubspace(C0, B).form(...)` |
    | discrete Dirichlet solve | `harmonic_extension(metric, boundary, prepare_direct)` |
    | vertex quadrature | positive degree-zero Hodge weights |

    The canonical `CochainSubspace` binds the ordered boundary indices to the exact
    degree-zero parent space. No disk evidence is assumed: the domain has two boundary
    components.
    """)
    return


@app.cell
def _(mo):
    resolution = mo.ui.slider(
        3, 6, value=4, step=1, label="radial rings (angular sections = 4 × rings)"
    )
    inner_value = mo.ui.slider(
        -2.0, 2.0, value=0.0, step=0.25, label="inner boundary value"
    )
    outer_value = mo.ui.slider(
        -2.0, 2.0, value=1.0, step=0.25, label="outer boundary value"
    )
    mo.vstack([resolution, inner_value, outer_value])
    return inner_value, outer_value, resolution


@app.cell
def _(
    CochainSubspace,
    ORDINARY_FORM,
    PositiveHodgeMetric,
    annulus,
    harmonic_extension,
    inner_value,
    np,
    outer_value,
    plot_cochain,
    prepare_direct,
    resolution,
    topological_boundary,
):
    def solve_study(rings):
        domain, geometry = annulus(rings, 4 * rings)
        metric = PositiveHodgeMetric(geometry)
        radii = np.linalg.norm(geometry.positions, axis=1)
        radius_inner = float(np.min(radii))
        radius_outer = float(np.max(radii))
        analytic = inner_value.value + (outer_value.value - inner_value.value) * np.log(
            radii / radius_inner
        ) / np.log(radius_outer / radius_inner)
        boundary_indices = np.flatnonzero(topological_boundary(domain).mask(0)).astype(
            np.int64
        )
        boundary_space = CochainSubspace(domain.cochain_space(0), boundary_indices)
        boundary = boundary_space.form(analytic[boundary_indices], ORDINARY_FORM)
        solution = harmonic_extension(metric, boundary, prepare_direct)
        values = solution.form.coefficients()
        derivative = domain.boundary_matrix(1).toarray()
        stiffness = derivative @ np.diag(metric.weights(1)) @ derivative.T
        interior_indices = np.setdiff1d(
            np.arange(domain.simplex_count(0)), boundary_indices
        )
        boundary_error = float(
            np.max(
                np.abs(values[boundary_indices] - boundary.coefficients()),
                initial=0.0,
            )
        )
        interior_residual = float(
            np.max(np.abs((stiffness @ values)[interior_indices]), initial=0.0)
        )
        error = values - analytic
        vertex_weights = metric.weights(0)
        weighted_l2_error = float(np.sqrt(np.sum(vertex_weights * error * error)))
        return (
            domain,
            geometry,
            metric,
            analytic,
            boundary,
            solution,
            boundary_error,
            interior_residual,
            weighted_l2_error,
        )

    study = solve_study(resolution.value)
    study_domain = study[0]
    study_geometry = study[1]
    study_metric = study[2]
    analytic_values = study[3]
    study_boundary = study[4]
    study_solution = study[5]
    exact_boundary_max = study[6]
    exact_interior_max = study[7]
    selected_l2_error = study[8]
    numerical_values = study_solution.form.coefficients()
    pointwise_error = np.abs(numerical_values - analytic_values)
    boundary_figure = plot_cochain(
        study_geometry, study_boundary, title="Canonical annulus boundary data"
    )
    solution_figure = plot_cochain(
        study_geometry, study_solution.form, title="Discrete harmonic extension"
    )
    error_form = study_domain.cochain_space(0).form(pointwise_error, ORDINARY_FORM)
    error_figure = plot_cochain(
        study_geometry, error_form, title="Absolute forward error |u − u★|"
    )

    convergence_rows = []
    previous_error = None
    for convergence_rings in range(3, 7):
        convergence_study = solve_study(convergence_rings)
        convergence_error = convergence_study[8]
        contraction = (
            "—"
            if previous_error is None
            else f"{previous_error / convergence_error:.3f}"
        )
        convergence_rows.append(
            (
                convergence_rings,
                4 * convergence_rings,
                convergence_study[0].simplex_count(0),
                convergence_error,
                contraction,
            )
        )
        previous_error = convergence_error
    return (
        boundary_figure,
        convergence_rows,
        error_figure,
        exact_boundary_max,
        exact_interior_max,
        selected_l2_error,
        solution_figure,
        study_domain,
        study_metric,
    )


@app.cell
def _(
    Complex,
    CochainSubspace,
    Geometry,
    ORDINARY_FORM,
    PositiveHodgeMetric,
    harmonic_extension,
    np,
):
    endpoint_domain = (
        Complex.from_maximal_simplices(np.array([[0, 1, 2]], dtype=np.int64))
        .codimension_one_regular()
        .with_boundary()
        .connected()
    )
    endpoint_geometry = Geometry.from_positions(
        endpoint_domain,
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]],
            dtype=np.float64,
        ),
    )
    endpoint_coefficients = np.array([-1.0, 0.5, 2.0], dtype=np.float64)
    endpoint_boundary = CochainSubspace(
        endpoint_domain.cochain_space(0), np.arange(3, dtype=np.int64)
    ).form(endpoint_coefficients, ORDINARY_FORM)
    backend_calls = [0]

    def forbidden_prepare(operator):
        backend_calls[0] += 1
        raise AssertionError(operator)

    endpoint_solution = harmonic_extension(
        PositiveHodgeMetric(endpoint_geometry),
        endpoint_boundary,
        forbidden_prepare,
    )
    endpoint_error = float(
        np.max(
            np.abs(endpoint_solution.form.coefficients() - endpoint_coefficients),
            initial=0.0,
        )
    )
    endpoint_backend_calls = backend_calls[0]
    endpoint_equation_size = endpoint_solution.equation_space.size
    return endpoint_backend_calls, endpoint_equation_size, endpoint_error


@app.cell
def _(inner_value, mo, outer_value, resolution, study_domain):
    mo.md(f"""
    ## Computation

    The selected deterministic mesh has **{resolution.value} radial rings**,
    **{4 * resolution.value} angular sections**, and
    **{study_domain.simplex_count(0)} vertices**. Its boundary values are
    $u_{{in}}={inner_value.value:.2f}$ and $u_{{out}}={outer_value.value:.2f}$.
    Resolution is deliberately bounded to the positive-Hodge-admitted fixture family.

    ## Visualization

    Boundary data, the completed field, and absolute analytic error are shown vertically.
    """)
    return


@app.cell
def _(boundary_figure):
    boundary_figure
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
    convergence_rows,
    endpoint_backend_calls,
    endpoint_equation_size,
    endpoint_error,
    exact_boundary_max,
    exact_interior_max,
    mo,
    selected_l2_error,
    study_metric,
):
    convergence_table = "\n".join(
        f"| {rings} | {sections} | {vertices} | {error:.6e} | {ratio} |"
        for rings, sections, vertices, error, ratio in convergence_rows
    )
    minimum_weight = min(
        float(min(study_metric.weights(degree))) for degree in range(3)
    )
    mo.md(f"""
    ## Evaluation

    | Independent law | Observed |
    |---|---:|
    | exact annulus boundary max | `{exact_boundary_max:.3e}` |
    | interior equation residual max | `{exact_interior_max:.3e}` |
    | selected weighted $L^2$ forward error | `{selected_l2_error:.6e}` |
    | minimum represented Hodge weight | `{minimum_weight:.3e}` |
    | all-boundary restoration error | `{endpoint_error:.3e}` |
    | all-boundary equation-space size | `{endpoint_equation_size}` |
    | all-boundary backend calls | `{endpoint_backend_calls}` |

    The convergence column reports successive weighted-error contraction, not an asserted
    asymptotic order:

    | rings | sections | vertices | weighted $L^2$ error | previous / current |
    |---:|---:|---:|---:|---:|
    {convergence_table}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation

    The solver satisfies its discrete boundary and interior equations to binary64 solve
    scale. The weighted $L^2$ table is a separate forward comparison with the continuum
    logarithmic solution; it supplies convergence evidence over this bounded mesh family
    without turning residual size into an accuracy claim.

    The all-boundary simplex is the endpoint law: with no interior unknowns, the values are
    returned exactly and the linear backend is never prepared.
    """)
    return


if __name__ == "__main__":
    app.run()
