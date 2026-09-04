import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo.chain import analyze_integral_homology
    from polygeo.plot import form as plot_form
    from examples.support.meshes import annulus

    return analyze_integral_homology, annulus, mo, np, plot_form


@app.cell
def _(mo):
    mo.md(r"""
    # Hodge decomposition on an annulus

    ## Question and prerequisites

    How does a metric split one discrete one-form into exact, coexact, and harmonic
    parts, and how does topology predict their dimension before any metric solve?

    We use the chain/cochain distinction from the homology lesson, a positive
    diagonal discrete Hodge star, exterior differentiation, and finite-dimensional
    least squares. The target decomposition is

    \[
    \omega=d\alpha+\delta\beta+h.
    \]

    Its three summands live in the same degree-one cochain space but have
    different mathematical origins: a vertex potential, a face potential, and
    the orthogonal remainder left by both projections.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The metric constructs the adjoint

    Let \(C^k\) be the real \(k\)-cochains and let the positive diagonal matrix
    \(\star_k\) represent the discrete metric inner product

    \[
    \langle x,y\rangle_k=x^{\mathsf T}\star_k y,
    \qquad
    \lVert x\rVert_k^2=\langle x,x\rangle_k.
    \]

    Exterior differentiation \(d_{k-1}:C^{k-1}\to C^k\) is topological. Its
    metric adjoint \(\delta_k:C^k\to C^{k-1}\) is forced by requiring

    \[
    \langle d_{k-1}\eta,\xi\rangle_k
      =\langle\eta,\delta_k\xi\rangle_{k-1}
    \]

    for every \(\eta\in C^{k-1}\) and \(\xi\in C^k\). Computing both sides in
    coordinates gives

    \[
    \eta^{\mathsf T}d_{k-1}^{\mathsf T}\star_k\xi
      =\eta^{\mathsf T}\star_{k-1}\delta_k\xi,
    \]

    hence positivity and invertibility of \(\star_{k-1}\) produce

    \[
    \boxed{\delta_k
      =\star_{k-1}^{-1}d_{k-1}^{\mathsf T}\star_k}.
    \]

    Thus \(d\) remembers incidence, while \(\delta\) additionally depends on the
    metric. Because \(d_kd_{k-1}=0\), the displayed formula also gives
    \(\delta_k\delta_{k+1}=0\).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Two least-energy projections

    The exact component is the closest element of
    \(\operatorname{im}d_{k-1}\) to \(\omega\). Define

    \[
    E_{\mathrm{ex}}(\alpha)
      =\frac12\lVert\omega-d_{k-1}\alpha\rVert_k^2.
    \]

    Varying by \(\eta\in C^{k-1}\) and applying adjointness computes

    \[
    \begin{aligned}
    \left.\frac{d}{d\varepsilon}
      E_{\mathrm{ex}}(\alpha+\varepsilon\eta)\right|_{\varepsilon=0}
      &=-\langle d\eta,\omega-d\alpha\rangle_k\\
      &=-\langle\eta,\delta_k(\omega-d\alpha)\rangle_{k-1}.
    \end{aligned}
    \]

    Vanishing for every \(\eta\) yields the exact normal equation

    \[
    \delta_kd_{k-1}\alpha=\delta_k\omega.
    \]

    Similarly, the coexact component minimizes

    \[
    E_{\mathrm{coex}}(\beta)
      =\frac12\lVert\omega-\delta_{k+1}\beta\rVert_k^2.
    \]

    Variation by \(\gamma\in C^{k+1}\) gives

    \[
    -\langle\delta\gamma,\omega-\delta\beta\rangle_k
      =-\langle\gamma,d_k(\omega-\delta\beta)\rangle_{k+1},
    \]

    so its normal equation is

    \[
    d_k\delta_{k+1}\beta=d_k\omega.
    \]

    At degree zero, \(C^{-1}=\{0\}\) removes the nonexistent exact term. At top
    degree \(n\), \(C^{n+1}=\{0\}\) removes the nonexistent coexact term. Degree
    one on a triangle surface has both projections.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The remainder is harmonic and orthogonal

    Set

    \[
    h=\omega-d\alpha-\delta\beta.
    \]

    The exact normal equation, \(\delta(\omega-d\alpha)=0\), combines with
    \(\delta^2=0\) to give

    \[
    \delta h
      =\delta(\omega-d\alpha)-\delta^2\beta
      =0.
    \]

    The coexact normal equation and \(d^2=0\) similarly give

    \[
    dh
      =d(\omega-\delta\beta)-d^2\alpha
      =0.
    \]

    Hence \(h\in\ker d\cap\ker\delta\). Adjointness now proves all three pairings:

    \[
    \begin{aligned}
    \langle d\alpha,\delta\beta\rangle_k
      &=\langle d^2\alpha,\beta\rangle_{k+1}=0,\\
    \langle d\alpha,h\rangle_k
      &=\langle\alpha,\delta h\rangle_{k-1}=0,\\
    \langle\delta\beta,h\rangle_k
      &=\langle\beta,dh\rangle_{k+1}=0.
    \end{aligned}
    \]

    Expanding the squared norm and deleting those cross terms produces the
    Pythagorean identity

    \[
    \lVert\omega\rVert_k^2
      =\lVert d\alpha\rVert_k^2
       +\lVert\delta\beta\rVert_k^2
       +\lVert h\rVert_k^2.
    \]

    Reconstruction, closedness, coclosedness, pairwise orthogonality, and this
    energy balance are distinct claims and receive distinct checks below.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Topology predicts the harmonic channel

    For the finite weighted complex and boundary convention used here, discrete
    Hodge theory identifies harmonic one-cochains with real first cohomology.
    Therefore

    \[
    \dim\mathcal H^1=b_1,
    \]

    where \(b_1\) is the free rank of integral first homology. An annulus retracts
    to a circle, so the exact topological analysis should return \(b_1=1\) before
    the metric decomposition is solved.

    To excite all three channels, let an oriented edge have vector
    \(e_{ij}=p_j-p_i\) and midpoint \(m_{ij}\). At \(m=(x,y)\), declare

    \[
    \begin{aligned}
    G(x,y)
      &=\nabla(0.7x-0.4y+0.12x^2-0.12y^2),\\
    R(x,y)
      &=J\nabla(0.15x^2+0.5xy+0.15y^2),\\
    H(x,y)
      &=0.8\,\frac{(-y,x)}{x^2+y^2},
    \qquad
    J(a,b)=(-b,a).
    \end{aligned}
    \]

    Here \(G\) is a declared gradient, \(R\) is a rotated gradient, and \(H\) is
    locally the differential of angle but has nonzero circulation around the
    hole. The sampled one-cochain is the midpoint quadrature

    \[
    \omega_{ij}\approx
      \langle G(m_{ij})+R(m_{ij})+H(m_{ij}),e_{ij}\rangle.
    \]

    This interpretable continuous mixture motivates the probe. Midpoint sampling
    does not assert exact equality between its three vector summands and the
    eventual discrete components.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Pseudocode

    ~~~text
    construct one oriented annulus and its positive degree-one metric weights
    compute the free rank of integral first homology
    require the topological prediction of one harmonic direction

    for each canonically oriented edge:
        compute its midpoint and oriented edge vector
        evaluate the declared gradient, rotated gradient, and circulation fields
        assign their summed midpoint dot product to the one-cochain

    prepare and solve one weighted Hodge decomposition
    reconstruct exact + coexact + harmonic
    compute the full weighted Gram matrix and all component energies
    independently apply exterior derivative and codifferential to the remainder
    compare harmonic rank prediction, reconstruction, pairings, and energy balance
    render the source and three components from the one retained result
    ~~~
    """)
    return


@app.cell
def _(analyze_integral_homology, annulus, np):
    domain, geometry = annulus(4, 16)
    metric = geometry.metric()
    group = analyze_integral_homology(domain.chain_complex(), [1])[1]
    predicted_harmonic_rank = group.free_rank
    if predicted_harmonic_rank != 1 or group.torsion_orders:
        raise RuntimeError("annulus first-homology prediction failed")

    space = domain.binary64_cochain_space(1)
    positions = geometry.positions_numpy_copy()
    edges = domain.simplices_numpy_copy(1)
    edge_vectors = positions[edges[:, 1]] - positions[edges[:, 0]]
    midpoints = 0.5 * (positions[edges[:, 0]] + positions[edges[:, 1]])
    x, y = midpoints[:, 0], midpoints[:, 1]
    gradient_field = np.column_stack((0.7 + 0.24 * x, -0.4 - 0.24 * y))
    stream_gradient = np.column_stack((0.5 * y + 0.3 * x, 0.5 * x + 0.3 * y))
    rotated_gradient = np.column_stack((-stream_gradient[:, 1], stream_gradient[:, 0]))
    radius_squared = x * x + y * y
    circulation_field = 0.8 * np.column_stack((-y, x)) / radius_squared[:, None]
    source_values = np.einsum(
        "ij,ij->i",
        gradient_field + rotated_gradient + circulation_field,
        edge_vectors,
    )
    source = space.admit_numpy(source_values)

    problem = metric.hodge_decomposition(source)
    prepared = problem.prepare()
    result = prepared.solve(problem, prepared.workspace_for(problem))
    exact_form, coexact_form, harmonic_form = (
        result.exact,
        result.coexact,
        result.harmonic,
    )
    component_values = np.vstack(
        [
            form.coefficients_numpy_copy()
            for form in (exact_form, coexact_form, harmonic_form)
        ]
    )
    weights = metric.hodge_coefficients_numpy_copy(1)
    weighted_gram = (component_values * weights) @ component_values.T
    component_energies = np.diag(weighted_gram)
    source_energy = float(np.dot(weights * source_values, source_values))
    energy_fractions = component_energies / source_energy
    reconstruction = np.sum(component_values, axis=0)
    reconstruction_error = float(np.max(np.abs(reconstruction - source_values)))
    pairings = (
        float(weighted_gram[0, 1]),
        float(weighted_gram[0, 2]),
        float(weighted_gram[1, 2]),
    )
    pythagorean_error = abs(source_energy - float(np.sum(component_energies)))
    harmonic_derivative = space.exterior_derivative().apply(harmonic_form)
    harmonic_codifferential = metric.codifferential(1).apply(harmonic_form)
    closure_error = float(np.max(np.abs(harmonic_derivative.coefficients_numpy_copy())))
    coclosure_error = float(
        np.max(np.abs(harmonic_codifferential.coefficients_numpy_copy()))
    )

    evidence_limit = 1.0e-10
    energy_fraction_floor = 1.0e-3
    errors = (
        reconstruction_error,
        max(abs(value) for value in pairings),
        pythagorean_error,
        closure_error,
        coclosure_error,
        result.reconstruction_bound,
        result.orthogonality_bound,
    )
    if max(errors) > evidence_limit:
        raise RuntimeError("Hodge decomposition evidence exceeds the study limit")
    if float(np.min(energy_fractions)) <= energy_fraction_floor:
        raise RuntimeError("the mixed probe does not excite every Hodge component")

    hodge_evidence = {
        "simplex_counts": tuple(domain.simplex_count(k) for k in range(3)),
        "predicted_harmonic_rank": predicted_harmonic_rank,
        "component_energies": tuple(float(value) for value in component_energies),
        "energy_fractions": tuple(float(value) for value in energy_fractions),
        "weighted_pairings": pairings,
        "reconstruction_error": reconstruction_error,
        "native_reconstruction_bound": result.reconstruction_bound,
        "native_orthogonality_bound": result.orthogonality_bound,
        "closure_error": closure_error,
        "coclosure_error": coclosure_error,
        "pythagorean_error": pythagorean_error,
        "evidence_limit": evidence_limit,
        "energy_fraction_floor": energy_fraction_floor,
    }
    study_forms = (
        (source, "Source one-form"),
        (exact_form, "Exact component"),
        (coexact_form, "Coexact component"),
        (harmonic_form, "Harmonic component"),
    )
    return geometry, hodge_evidence, study_forms


@app.cell
def _(hodge_evidence, mo):
    component_names = ("exact", "coexact", "harmonic")
    energy_rows = "\n".join(
        f"| {name} | {energy:.6f} | {fraction:.3%} |"
        for name, energy, fraction in zip(
            component_names,
            hodge_evidence["component_energies"],
            hodge_evidence["energy_fractions"],
            strict=True,
        )
    )
    mo.md(rf"""
    ## 6. Numerical evidence

    The annulus has {hodge_evidence["simplex_counts"]} vertices, edges, and faces.
    Exact first homology predicts harmonic rank
    {hodge_evidence["predicted_harmonic_rank"]} before the metric solve.

    | Component | Weighted energy | Source-energy fraction |
    |:--|--:|--:|
    {energy_rows}

    | Certificate | Observed bound |
    |:--|--:|
    | Independent reconstruction | {hodge_evidence["reconstruction_error"]:.3e} |
    | Native reconstruction | {hodge_evidence["native_reconstruction_bound"]:.3e} |
    | Three independent weighted pairings | {hodge_evidence["weighted_pairings"]} |
    | Native exact/coexact orthogonality | {hodge_evidence["native_orthogonality_bound"]:.3e} |
    | Harmonic closedness | {hodge_evidence["closure_error"]:.3e} |
    | Harmonic coclosedness | {hodge_evidence["coclosure_error"]:.3e} |
    | Pythagorean energy balance | {hodge_evidence["pythagorean_error"]:.3e} |

    Every algebraic bound is at most
    {hodge_evidence["evidence_limit"]:.1e}. Every component carries more than
    {hodge_evidence["energy_fraction_floor"]:.1e} of the source energy. These are
    regression gates for this deterministic probe, not universal error estimates.
    """)
    return


@app.cell
def _(geometry, mo, plot_form, study_forms):
    component_figures = [
        plot_form(geometry, form, title=title) for form, title in study_forms
    ]
    mo.vstack(
        [
            mo.md(r"""
            ## 7. Source and retained components

            Every panel uses the same oriented edges. Color shows signed
            coefficients; geometry is context. The three component panels come
            from the one retained solve and are not independently refitted.
            """),
            mo.hstack(component_figures[:2]),
            mo.hstack(component_figures[2:]),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation and limits

    Topology predicts that one harmonic direction can survive both differential
    tests; the metric decides its least-energy representative and makes all three
    components orthogonal. The independent reconstruction and Gram matrix test
    the splitting, while applying \(d\) and \(\delta\) directly tests the
    harmonic remainder. None of those checks can substitute for another.

    The continuous vector mixture makes the discrete probe interpretable, but its
    midpoint edge evaluations are quadrature. The experiment does not identify
    the computed components pointwise with the three declared vector fields, prove
    mesh convergence, or establish the same numerical bounds for another
    triangulation or metric.
    """)
    return


if __name__ == "__main__":
    app.run()
