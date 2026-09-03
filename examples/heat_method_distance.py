import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo.geometry import TriangleSurface
    from polygeo.plot import form as plot_form
    from support.meshes import icosphere

    return TriangleSurface, icosphere, mo, np, plot_form


@app.cell
def _(mo):
    mo.md(r"""
    # Heat-method distance on a closed surface

    ## Question and prerequisites

    How can one short diffusion step, one normalized face field, and one Poisson
    solve approximate distance from a selected vertex?

    We use the positive-semidefinite convention
    $\Delta=-\operatorname{div}\operatorname{grad}$ from the Poisson lesson. We
    also use its vertex mass matrix $M$, stiffness matrix $K$, integrated loads,
    compatibility condition, and mean-zero gauge. The construction below keeps
    the diffusion solve, geometric approximation, and Poisson solve as distinct
    mathematical phases.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Diffuse an integrated impulse

    Let $u(t)$ solve the heat equation

    $$
    \partial_tu+\Delta u=0.
    $$

    A backward-Euler step of duration $t>0$ from $u_0$ evaluates the Laplacian at
    the unknown endpoint:

    $$
    \frac{u-u_0}{t}+\Delta u=0.
    $$

    Multiplying by $tM$ and using $M\Delta=K$ constructs the finite system

    $$
    M(u-u_0)+tKu=0,
    $$

    hence

    $$
    (M+tK)u=Mu_0=b_0.
    $$

    The right-hand side is an integrated source, not a pointwise sample. For one
    source vertex $s$, choose

    $$
    (u_0)_s=\frac{1}{m_s},
    \qquad
    (u_0)_i=0\quad(i\ne s).
    $$

    Then $b_0=Mu_0$ has coefficient one at $s$, zero elsewhere, and total
    integrated mass $\mathbf{1}^Tb_0=1$. For several selected vertices, dividing
    by their count distributes that unit mass uniformly among them.

    Since $M$ is positive definite and $K$ is positive semidefinite,

    $$
    z^T(M+tK)z=z^TMz+t\,z^TKz>0
    \qquad(z\ne0),
    $$

    so this heat system has a unique solution without a gauge condition.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Recover an outward direction

    Away from the source and its cut locus, the short-time heat kernel has the
    leading form

    $$
    u(t,x)\approx C(t,x)\exp\!\left(-\frac{d(x,s)^2}{4t}\right),
    $$

    where the prefactor varies more slowly than the exponential as $t$ becomes
    small. Differentiating the dominant logarithm gives

    $$
    \nabla\log u
      \approx-\frac{d}{2t}\nabla d.
    $$

    Thus $\nabla u$ points inward, toward the source. Its negative normalized
    direction

    $$
    X=-\frac{\nabla u}{\lVert\nabla u\rVert}
    $$

    approximates the outward distance gradient $\nabla d$ wherever the heat
    gradient is nonzero.

    ### One constant gradient per triangle

    On an oriented face $T=(x_0,x_1,x_2)$, let $\lambda_i$ be the barycentric
    basis functions, $A_T$ its area, and $n_T$ its unit normal. The linear heat
    interpolant is

    $$
    u_h|_T=\sum_{i=0}^2u_i\lambda_i.
    $$

    Its basis gradients are constant on the face:

    $$
    \nabla\lambda_0=\frac{n_T\times(x_2-x_1)}{2A_T},\qquad
    \nabla\lambda_1=\frac{n_T\times(x_0-x_2)}{2A_T},\qquad
    \nabla\lambda_2=\frac{n_T\times(x_1-x_0)}{2A_T}.
    $$

    Therefore the computed face gradient is the explicit linear combination

    $$
    g_T=\nabla u_h|_T=\sum_{i=0}^2u_i\nabla\lambda_i.
    $$

    It is a piecewise-constant approximation, not an exact smooth vector field.
    If $\lVert g_T\rVert=0$, its direction is mathematically undefined. This
    bounded experiment refuses that case instead of manufacturing a direction;
    otherwise it uses $X_T=-g_T/\lVert g_T\rVert$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Integrate divergence and recover a potential

    For a facewise vector field $X$, define its standard integrated divergence
    $q_X$ by the negative-adjoint identity

    $$
    v^Tq_X
      =-\sum_T A_T\,\langle\nabla v|_T,X_T\rangle
      \approx\int_S v\,\operatorname{div}X\,dA.
    $$

    The sign needed by the positive Laplacian follows by asking for
    $\nabla\phi\approx X$:

    $$
    v^TK\phi
      =\sum_T A_T\,\langle\nabla v|_T,\nabla\phi|_T\rangle
      \approx\sum_T A_T\,\langle\nabla v|_T,X_T\rangle
      =-v^Tq_X.
    $$

    Consequently the Poisson right-hand side is

    $$
    b_{\mathrm{div}}=-q_X,
    \qquad
    K\phi=b_{\mathrm{div}}.
    $$

    Equivalently, let $Y=-X=\nabla u/\lVert\nabla u\rVert$ be the inward heat
    direction. Linearity gives $q_Y=-q_X=b_{\mathrm{div}}$, so taking the standard
    integrated divergence of the inward field constructs exactly the required
    positive-Laplacian load. This is the convention used in the experiment.

    Compatibility is automatic: inserting the constant test vector gives

    $$
    \mathbf{1}^Tq_Y
      =-\sum_T A_T\,\langle\nabla1,Y_T\rangle=0.
    $$

    The Poisson solution is determined only up to a constant. A weighted
    mean-zero solve chooses an internal representative, after which

    $$
    d_h=\phi-\min_i\phi_i
    $$

    makes the smallest reported distance zero. This final shift changes neither
    face gradients nor the Poisson equation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Choose the diffusion scale and classify evidence

    Heat propagates over a characteristic length proportional to $\sqrt{t}$.
    Let $h$ be the mean mesh-edge length. Choosing

    $$
    t=h^2
    $$

    makes the diffusion length comparable to one representative edge. Under a
    uniform spatial scaling $x\mapsto\alpha x$, both $h$ and intrinsic distance
    scale by $\alpha$, while $t$ scales by $\alpha^2$, as diffusion requires.

    Three evidence classes answer different questions:

    1. Heat and Poisson residual bounds certify their assembled linear systems.
    2. On the unit sphere, the analytic reference
       $d_{\mathrm{sph}}(x,s)=\arccos(\widehat{x}\mathbin{\cdot}\widehat{s})$
       measures geometric approximation error.
    3. The mean facewise Eikonal defect

       $$
       E_{\mathrm{eik}}
         =\frac{1}{F}\sum_T
            \left|\lVert\nabla d_h|_T\rVert-1\right|
       $$

       measures how closely the recovered field satisfies the unit-gradient law.

    A small solver residual does not imply small spherical error, and neither
    quantity implies a small Eikonal defect. Comparing two fixed resolutions can
    reveal an observed trend, but it is not a convergence proof.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Algorithm

    ```text
    for each selected mesh resolution:
        construct the unit sphere once
        h := mean edge length
        t := h*h
        u0 := mass-normalized impulse at the source vertex
        b0 := M u0
        solve (M + tK) u = b0

        for each face:
            g := gradient of the linear heat field on that face
            refuse the experiment if length(g) is zero
            X := -g / length(g)

        q_X := standard integrated divergence of X
        b_div := -q_X
        solve K phi = b_div with a weighted mean-zero gauge
        distance := phi - minimum(phi)

        compare distance with great-circle distance
        measure both linear residuals and the mean Eikonal defect

    reuse the already-computed finest distance for the single figure
    render one row of evidence per resolution
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Experiment

    Two deterministic unit-sphere meshes use the same source vertex and the same
    scale rule. Each resolution performs exactly one heat preparation and solve,
    followed by exactly one compatible Poisson preparation and solve. The finer
    result is retained for the figure rather than recomputed.
    """)
    return


@app.cell
def _(TriangleSurface, np):
    def heat_method_distance(geometry, source_indices):
        if (
            source_indices.ndim != 1
            or source_indices.size == 0
            or source_indices.dtype.kind not in "iu"
            or np.unique(source_indices).size != source_indices.size
            or np.any(source_indices < 0)
            or np.any(source_indices >= geometry.topology.vertex_count)
        ):
            raise ValueError("sources must be nonempty unique vertex indices")

        metric = geometry.metric()
        surface = TriangleSurface.admit(geometry)
        space = geometry.topology.binary64_cochain_space(0)
        masses = metric.hodge_coefficients_numpy_copy(0)
        initial = np.zeros(geometry.topology.vertex_count, dtype=np.float64)
        initial[source_indices] = 1.0 / (source_indices.size * masses[source_indices])

        edge_lengths = geometry.primal_measures_numpy_copy(1)
        length_scale = np.max(edge_lengths)
        mean_edge_length = length_scale * np.mean(edge_lengths / length_scale)
        time_step = mean_edge_length * mean_edge_length
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("mean edge length does not define a positive finite time")

        heat_problem = metric.heat_evolution(space.admit_numpy(initial), time_step)
        heat_prepared = heat_problem.prepare()
        heat = heat_prepared.solve(
            heat_problem, heat_prepared.workspace_for(heat_problem)
        )

        inward_direction = surface.gradient(heat.value).normalized()
        poisson_load = surface.divergence(inward_direction)
        poisson_problem = metric.mean_zero_poisson_load(poisson_load)
        poisson_prepared = poisson_problem.prepare()
        poisson = poisson_prepared.solve(
            poisson_problem, poisson_prepared.workspace_for(poisson_problem)
        )

        potential = poisson.potential.coefficients_numpy_copy()
        coefficients = potential - np.min(potential)
        distance = space.admit_numpy(coefficients)
        gradient_lengths = np.linalg.norm(
            surface.gradient(distance).values_numpy_copy(), axis=1
        )
        evidence = {
            "mean_edge_length": float(mean_edge_length),
            "time_step": float(time_step),
            "heat_residual_bound": heat.residual_bound,
            "poisson_residual_bound": poisson.residual_bound,
            "mean_eikonal_defect": float(np.mean(np.abs(gradient_lengths - 1.0))),
        }
        return distance, evidence

    return (heat_method_distance,)


@app.cell
def _(heat_method_distance, icosphere, np):
    source = 0
    study_rows = []
    study_outputs = []
    for subdivision in (1, 2):
        _, geometry = icosphere(subdivision, 1.0)
        distance, solve_evidence = heat_method_distance(
            geometry, np.array([source], dtype=np.int64)
        )
        coefficients = distance.coefficients_numpy_copy()
        positions = geometry.positions_numpy_copy()
        unit_positions = positions / np.linalg.norm(positions, axis=1)[:, None]
        analytic_distance = np.arccos(
            np.clip(unit_positions @ unit_positions[source], -1.0, 1.0)
        )
        spherical_error = np.abs(coefficients - analytic_distance)
        if int(np.argmin(coefficients)) != source:
            raise RuntimeError("the selected source is not the distance minimum")
        study_rows.append(
            solve_evidence
            | {
                "subdivision": subdivision,
                "vertex_count": geometry.topology.vertex_count,
                "maximum_spherical_error": float(np.max(spherical_error)),
                "mean_spherical_error": float(np.mean(spherical_error)),
            }
        )
        study_outputs.append((geometry, distance))

    coarse_evidence, fine_evidence = study_rows
    if not (
        fine_evidence["maximum_spherical_error"]
        < coarse_evidence["maximum_spherical_error"]
        and fine_evidence["mean_spherical_error"]
        < coarse_evidence["mean_spherical_error"]
        and fine_evidence["mean_eikonal_defect"]
        < coarse_evidence["mean_eikonal_defect"]
    ):
        raise RuntimeError("the two-resolution approximation trend was not observed")
    if fine_evidence["maximum_spherical_error"] > 0.12:
        raise RuntimeError("heat-method error exceeds the sphere-study bound")
    if fine_evidence["mean_eikonal_defect"] > 0.04:
        raise RuntimeError("heat-method eikonal defect exceeds the study bound")

    finest_geometry, finest_distance = study_outputs[-1]
    return finest_distance, finest_geometry, study_rows


@app.cell
def _(mo, study_rows):
    solve_rows = "\n".join(
        f"| {int(row['subdivision'])} | {int(row['vertex_count'])} | "
        f"`{row['mean_edge_length']:.4f}` | `{row['time_step']:.4f}` | "
        f"`{row['heat_residual_bound']:.3e}` | "
        f"`{row['poisson_residual_bound']:.3e}` |"
        for row in study_rows
    )
    geometry_rows = "\n".join(
        f"| {int(row['subdivision'])} | "
        f"`{row['maximum_spherical_error']:.4f}` | "
        f"`{row['mean_spherical_error']:.4f}` | "
        f"`{row['mean_eikonal_defect']:.4f}` |"
        for row in study_rows
    )
    mo.md(f"""
    ## Evidence

    Each row comes from one completed operator composition. Residual bounds are
    algebraic certificates; the final three columns are geometric diagnostics.

    ### Scale and solve certificates

    | Level | Vertices | Mean edge $h$ | $t=h^2$ | Heat residual | Poisson residual |
    |---:|---:|---:|---:|---:|---:|
    {solve_rows}

    ### Geometric diagnostics

    | Level | Max sphere error | Mean sphere error | Mean Eikonal defect |
    |---:|---:|---:|---:|
    {geometry_rows}

    Both spherical-error summaries and the mean Eikonal defect are smaller on the
    finer mesh. This is the observed behavior of these two fixed experiments, not
    a claim of monotone convergence for arbitrary meshes or time-step choices.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualization

    The single figure reuses the computed level-two distance. Color encodes the
    shifted vertex values $d_h$, with zero at the selected source and increasing
    values away from it. A sequential color scale begins at zero, so no unused
    negative range is implied. The labeled colorbar reports distance magnitude;
    the neutral mesh supplies geometric context.
    """)
    return


@app.cell
def _(finest_distance, finest_geometry, np, plot_form):
    distance_figure = plot_form(
        finest_geometry, finest_distance, title="Heat-method distance"
    )
    maximum_distance = float(np.max(finest_distance.coefficients_numpy_copy()))
    distance_figure.update_traces(
        hovertemplate=(
            "vertex %{customdata[0]}<br>distance %{customdata[1]}<extra></extra>"
        ),
        marker={
            "cmin": 0.0,
            "cmax": maximum_distance,
            "colorbar": {"title": {"text": "distance"}},
            "colorscale": "Viridis",
        },
        name="distance",
        selector={"name": "form"},
    )
    distance_figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation and limits

    The construction links four semantic objects: an integrated impulse produces
    a short-time heat field; its inward gradient determines an outward direction;
    the signed integrated divergence becomes a compatible Poisson load; and a
    constant shift selects nonnegative reported distances.

    The small linear residuals show that the two algebraic systems were solved
    consistently. The sphere errors instead compare with a known smooth distance,
    while the Eikonal defect checks a local unit-gradient law. Their simultaneous
    decrease under this one refinement is encouraging but does not establish
    convergence, exact polyhedral shortest paths, behavior at the cut locus, or
    accuracy on arbitrary meshes. Those claims require a broader refinement study
    and separate geometric analysis.
    """)
    return


if __name__ == "__main__":
    app.run()
