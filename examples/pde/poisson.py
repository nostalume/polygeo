import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo.plot import form as plot_form
    from examples.support.meshes import icosphere

    return icosphere, mo, np, plot_form


@app.cell
def _(mo):
    mo.md(r"""
    # Mean-zero Poisson problem

    ## Question and prerequisites

    How does a pointwise source density become the right-hand side of a Poisson
    equation on a closed triangle surface?

    We build on the preceding distinction between a vertex-supported measure and
    a density. We assume finite-dimensional linear algebra, oriented edges, and
    the weak form of integration by parts. The outcome will be two equivalent
    descriptions of one compatible source and one mean-zero potential.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Mathematical model

    Let $S$ be a smooth, closed, connected surface. Choose the positive
    semidefinite sign convention

    $$
    \Delta=-\operatorname{div}\operatorname{grad}.
    $$

    Given a source density $f$, the equation $\Delta u=f$ is tested against every
    smooth function $v$. Because $S$ has no boundary, integration by parts gives

    $$
    \int_S v\,\Delta u\,dA
      =\int_S \langle\nabla u,\nabla v\rangle\,dA.
    $$

    The weak Poisson problem is therefore

    $$
    \int_S \langle\nabla u,\nabla v\rangle\,dA
      =\int_S f v\,dA
      \qquad\text{for every }v.
    $$

    Setting $v=1$ makes the left side zero, so a solution requires

    $$
    \int_S f\,dA=0.
    $$

    Adding a constant to $u$ does not change its gradient. We select one solution
    from this affine family by imposing the mean-zero gauge

    $$
    \int_S u\,dA=0.
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Discrete model

    Let $u,v\in\mathbb{R}^V$ store piecewise-linear values at the vertices of an
    oriented triangle mesh. The vertex-to-edge coboundary $d_0$ computes an
    oriented difference on every edge. With a positive diagonal edge Hodge star
    $\star_1$, the discrete Dirichlet pairing is

    $$
    \langle d_0u,d_0v\rangle_{\star_1}
      =(d_0v)^T\star_1(d_0u)
      =v^T\underbrace{d_0^T\star_1d_0}_{K}u.
    $$

    This constructs the stiffness matrix

    $$
    K=d_0^T\star_1d_0.
    $$

    Its sign follows from the same factorization:

    $$
    u^TKu=(d_0u)^T\star_1(d_0u)\geq 0.
    $$

    Next let $M=\star_0=\operatorname{diag}(m_0,\ldots,m_{V-1})$ be the positive
    vertex mass matrix. A vector $f$ stores density samples, whereas

    $$
    b=Mf
    $$

    stores their integrated action on the vertex test functions. Although $f$
    and $b$ both have $V$ entries, they represent different mathematical
    objects. Substitution in the weak equation gives

    $$
    v^TKu=v^TMf\quad\text{for every }v,
    $$

    and therefore

    $$
    Ku=Mf=b.
    $$

    The strong discrete Laplacian is the mass-normalized operator

    $$
    \Delta_h=M^{-1}K,
    $$

    so the two descriptions have an explicit coincidence witness:

    $$
    \Delta_hu=f
      \quad\Longleftrightarrow\quad
    Ku=Mf=b.
    $$

    ### Compatibility and nullspace

    The constant vector $\mathbf{1}$ satisfies $d_0\mathbf{1}=0$, hence
    $K\mathbf{1}=0$. Symmetry also gives $\mathbf{1}^TK=0$. Multiplying the load
    equation by $\mathbf{1}^T$ yields the necessary compatibility law

    $$
    \mathbf{1}^Tb=0,
    \qquad\text{equivalently}\qquad
    \mathbf{1}^TMf=0.
    $$

    For a connected mesh with positive $\star_1$, only constants have zero
    Dirichlet energy, so $\ker K=\operatorname{span}\{\mathbf{1}\}$. Compatibility
    gives existence modulo constants; the weighted gauge

    $$
    \mathbf{1}^TMu=0
    $$

    chooses a unique representative.

    In the experiment, two distinct vertices $p$ and $q$ receive

    $$
    f_p=m_q,
    \qquad
    f_q=-m_p,
    \qquad
    f_i=0\text{ otherwise}.
    $$

    The construction is compatible before solving because

    $$
    \mathbf{1}^TMf=m_pm_q-m_qm_p=0.
    $$

    On a two-dimensional surface, $m_i$ has units of area. If $u$ has units
    $[U]$, then $f$ has units $[U]/L^2$ and $b=Mf$ has units $[U]$, matching
    $Ku$. The algebraic residuals below certify the finite linear system; without
    mesh refinement or comparison to an exact solution, they do not measure
    approximation error to a smooth physical field.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Algorithm

    ```text
    construct one closed connected triangle mesh
    assemble vertex masses M and stiffness K

    choose distinct vertices p and q
    set f[p] := M[q,q]
    set f[q] := -M[p,p]
    set every other density value to zero
    b := M f
    verify sum(b) = 0

    factorize one nonsingular representative of K modulo constants
    allocate one compatible solve workspace
    u_density := solve K u = M f with weighted mean zero
    u_load := solve K u = b with weighted mean zero

    reapplied_load := K u_load
    report:
        infinity_norm(b - M f)
        infinity_norm(reapplied_load - b)
        infinity_norm(u_density - u_load)
        absolute_value(1^T M u_load)
        maximum(u_load) - minimum(u_load)
    render f and u_load
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Experiment

    A deterministic closed sphere supplies one positive vertex mass matrix. The
    source uses two vertices and is compatible by construction. We build the mesh,
    metric, density, and integrated load once.
    """)
    return


@app.cell
def _(icosphere, np):
    domain, geometry = icosphere(1, 1.0)
    vertex_space = domain.binary64_cochain_space(0)
    metric = geometry.metric()
    masses = metric.hodge_coefficients_numpy_copy(0)
    density_values = np.zeros(vertex_space.size, dtype=np.float64)
    density_values[:2] = (masses[1], -masses[0])
    density = vertex_space.admit_numpy(density_values)
    load = metric.riesz(0).apply(density)
    return density, geometry, load, masses, metric


@app.cell
def _(density, load, metric):
    density_problem = metric.mean_zero_poisson_density(density)
    load_problem = metric.mean_zero_poisson_load(load)
    return density_problem, load_problem


@app.cell
def _(density_problem, load_problem):
    prepared = density_problem.prepare()
    workspace = prepared.workspace_for(load_problem)
    return prepared, workspace


@app.cell
def _(density_problem, load_problem, prepared, workspace):
    density_solution = prepared.solve(density_problem, workspace)
    load_solution = prepared.solve(load_problem, workspace)
    potential_form = load_solution.potential
    return density_solution, load_solution, potential_form


@app.cell
def _(density, density_solution, load, masses, metric, mo, np, potential_form):
    potential = potential_form.coefficients_numpy_copy()
    density_copy = density.coefficients_numpy_copy()
    load_values = load.coefficients_numpy_copy()
    mass_action_error = float(np.max(np.abs(load_values - masses * density_copy)))
    reapplied_load = (
        metric.riesz(0)
        .apply(metric.laplacian(0).apply(potential_form))
        .coefficients_numpy_copy()
    )
    load_compatibility = float(abs(np.sum(load_values)))
    load_residual = float(np.max(np.abs(reapplied_load - load_values)))
    density_load_solution_agreement = float(
        np.max(np.abs(density_solution.potential.coefficients_numpy_copy() - potential))
    )
    weighted_gauge = float(abs(masses @ potential))
    potential_range = float(np.ptp(potential))
    evidence_limit = 1.0e-12
    if not (
        mass_action_error <= evidence_limit
        and load_compatibility <= evidence_limit
        and load_residual <= evidence_limit
        and density_load_solution_agreement <= evidence_limit
        and weighted_gauge <= evidence_limit
        and potential_range > 0.0
    ):
        raise RuntimeError("Poisson-study evidence exceeds its declared limits")

    mo.md(rf"""
    ## Evidence

    The load is formed independently as $b=Mf$, while the reapplied operator is
    evaluated as $M\Delta_hu_{{\mathrm{{load}}}}=Ku_{{\mathrm{{load}}}}$. All
    algebraic certificates use the absolute tolerance `{evidence_limit:.1e}`.

    | Certificate | Observed | Required claim | Result |
    |---|---:|---:|---|
    | Mass action $\lVert b-Mf\rVert_\infty$ | `{mass_action_error:.3e}` | $\leq {evidence_limit:.1e}$ | `{mass_action_error <= evidence_limit}` |
    | Load compatibility $\lvert\mathbf{{1}}^Tb\rvert$ | `{load_compatibility:.3e}` | $\leq {evidence_limit:.1e}$ | `{load_compatibility <= evidence_limit}` |
    | Backward residual $\lVert Ku_{{\mathrm{{load}}}}-b\rVert_\infty$ | `{load_residual:.3e}` | $\leq {evidence_limit:.1e}$ | `{load_residual <= evidence_limit}` |
    | Density/load solution agreement | `{density_load_solution_agreement:.3e}` | $\leq {evidence_limit:.1e}$ | `{density_load_solution_agreement <= evidence_limit}` |
    | Weighted gauge $\lvert\mathbf{{1}}^TMu_{{\mathrm{{load}}}}\rvert$ | `{weighted_gauge:.3e}` | $\leq {evidence_limit:.1e}$ | `{weighted_gauge <= evidence_limit}` |
    | Potential range | `{potential_range:.6f}` | $>0$ | `{potential_range > 0.0}` |

    The first row computes the density-to-load transformation by both routes. The
    next two check solvability and the integrated-load equation; the fourth checks
    coincidence of the resulting potentials; and the fifth checks the chosen
    representative. These are algebraic certificates, not forward
    physical-accuracy estimates.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Visualization

    Both figures place values at vertices. In the first, color encodes the signed
    density samples $f_i$; in the second, color encodes the signed mean-zero
    potential $u_i$. Each labeled colorbar gives the numerical coefficient scale,
    while the neutral mesh provides geometric context. Color magnitudes from the
    two figures should not be compared as if they had the same units.
    """)
    return


@app.cell
def _(density, geometry, mo, plot_form, potential_form):
    density_figure = plot_form(geometry, density, title="Compatible density")
    potential_figure = plot_form(geometry, potential_form, title="Mean-zero potential")
    mo.hstack([density_figure, potential_figure])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation and limits

    A density is not inserted directly into the weak stiffness equation: the mass
    matrix first turns it into an integrated load. The identities
    $\Delta_h=M^{-1}K$ and $b=Mf$ explain why the density and load formulations
    produce the same weighted mean-zero potential.

    Compatibility removes the obstruction created by the constant nullspace, and
    the weighted gauge removes the remaining additive ambiguity. The small
    residuals establish that this finite system was solved consistently. They do
    not establish convergence under refinement or accuracy for a chosen continuum
    source. The heat-distance study composes mass, stiffness, gradient,
    normalization, and Poisson operations into a distance approximation.
    """)
    return


if __name__ == "__main__":
    app.run()
