import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from polygeo.chain import analyze_integral_homology
    from polygeo.plot import homology_cycle as plot_homology_cycle
    from examples.support.meshes import torus

    return analyze_integral_homology, mo, plot_homology_cycle, torus


@app.cell
def _(mo):
    mo.md(r"""
    # Integral homology on a triangulated torus

    ## Question and prerequisites

    How does a signed simplicial boundary produce a quotient group of cycles,
    how is its algebraic dual related by Stokes' identity, and why does a
    unimodular cup pairing detect the torus's two independent directions?

    We use finite oriented simplices and integer coefficients. No metric,
    floating-point tolerance, or embedding is needed for the topology. The
    Euclidean torus appears only after every algebraic claim has been checked,
    when two selected cycle representatives are drawn. No earlier study is
    required.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Oriented chains force a chain complex

    For each degree \(k\), the free abelian chain group is

    \[
    C_k(K;\mathbb Z)
      =\left\{\sum_{\sigma\in K_k}n_\sigma[\sigma]\;:\;
        n_\sigma\in\mathbb Z,\ \text{finitely supported}\right\}.
    \]

    Reversing a simplex orientation negates its generator. On a canonically
    ordered simplex, define the signed boundary by deleting one vertex at a time:

    \[
    \partial_k[v_0,\ldots,v_k]
      =\sum_{i=0}^{k}(-1)^i
        [v_0,\ldots,\widehat v_i,\ldots,v_k].
    \]

    The smallest nontrivial cancellation occurs on an oriented face:

    \[
    \begin{aligned}
    \partial_2[v_0,v_1,v_2]
      &=[v_1,v_2]-[v_0,v_2]+[v_0,v_1],\\
    \partial_1\partial_2[v_0,v_1,v_2]
      &=([v_2]-[v_1])-([v_2]-[v_0])+([v_1]-[v_0])\\
      &=0.
    \end{aligned}
    \]

    In every degree, each codimension-two face is obtained by deleting two
    vertices in two orders. Those two terms have opposite signs, so
    \(\partial_{k-1}\partial_k=0\). This identity is the obstruction-removing
    fact that makes the quotient below possible.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Homology remembers cycles modulo filled cycles

    Two subgroups of \(C_k\) arise from the chain-complex identity:

    \[
    Z_k=\ker\partial_k,
    \qquad
    B_k=\operatorname{im}\partial_{k+1}.
    \]

    If \(b\in B_k\), then \(b=\partial_{k+1}c\) for some \(c\), and hence

    \[
    \partial_k b
      =\partial_k\partial_{k+1}c
      =0.
    \]

    Thus \(B_k\subseteq Z_k\), and the quotient

    \[
    H_k(K;\mathbb Z)=Z_k/B_k
    \]

    identifies closed chains that differ by the boundary of a higher chain.
    A class \([z]\) is therefore not the same object as a chosen cycle \(z\).
    Sparse representatives are useful witnesses and useful drawing inputs, but
    their term order and shape may change after relabeling or another valid basis
    choice.

    The structure theorem separates invariant information:

    \[
    H_k\cong
      \mathbb Z^{\beta_k}
      \oplus\bigoplus_j\mathbb Z/\tau_{k,j}\mathbb Z,
      \qquad \tau_{k,j}>1.
    \]

    Here \(\beta_k\) is the free rank and the \(\tau_{k,j}\) are torsion orders.
    The experiment reports those invariants separately from selected generators.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Algebraic duality constructs discrete Stokes

    Integer \(k\)-cochains are homomorphisms from chains to integers:

    \[
    C^k(K;\mathbb Z)=\operatorname{Hom}(C_k(K;\mathbb Z),\mathbb Z),
    \qquad
    \langle\alpha,c\rangle=\alpha(c).
    \]

    The coboundary is forced by asking the dual operation to consume the boundary
    of the same chain. For \(\alpha\in C^k\) and \(c\in C_{k+1}\), define
    \(d^k\alpha=\alpha\circ\partial_{k+1}\). Evaluating both composites gives

    \[
    \langle d^k\alpha,c\rangle
      =(\alpha\circ\partial_{k+1})(c)
      =\alpha(\partial_{k+1}c)
      =\langle\alpha,\partial_{k+1}c\rangle.
    \]

    This is the exact chain/cochain Stokes identity. Moreover,

    \[
    d^{k+1}d^k\alpha
      =\alpha\circ\partial_{k+1}\circ\partial_{k+2}
      =0,
    \]

    so cocycles \(\ker d\) also form a complex. The experiment chooses one
    edge cochain and one face whose two Stokes evaluations are equal and nonzero;
    zero would check the syntax but would not expose the sign.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Cup product turns dual cycles into an intersection witness

    For a \(p\)-cochain \(\alpha\), a \(q\)-cochain \(\beta\), and an oriented
    \((p+q)\)-simplex, the Alexander--Whitney rule splits the simplex at a shared
    vertex:

    \[
    (\alpha\smile\beta)[v_0,\ldots,v_{p+q}]
      =\alpha[v_0,\ldots,v_p]\,
       \beta[v_p,\ldots,v_{p+q}].
    \]

    Let \(a,b\in C^1\) be closed and let \(z\in Z_2\) represent an oriented
    fundamental class. Graded commutativity is a statement about cohomology
    classes,

    \[
    [a]\smile[b]=-[b]\smile[a]\quad\text{in }H^2,
    \]

    not an assertion that the two raw degree-two cochains are negatives. Their
    difference is a coboundary; pairing it with \(z\) vanishes by Stokes because
    \(\partial z=0\). Therefore

    \[
    \langle a\smile b,z\rangle
      =-\langle b\smile a,z\rangle.
    \]

    For two torus generators the experiment obtains a pairing matrix of the form

    \[
    Q=\begin{pmatrix}0&s\\-s&0\end{pmatrix},
    \qquad |s|=1,
    \qquad \det Q=s^2=1.
    \]

    Its determinant is a unit in \(\mathbb Z\), so the pairing identifies the two
    selected free directions with an integral dual basis. Reordering or negating
    representatives may change entries and signs, but not unimodularity.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Pseudocode

    ~~~text
    construct one oriented torus chain complex over the integers
    choose one oriented face
    verify boundary(boundary(face)) is the zero chain

    choose one edge cochain with nonzero evaluation on that face boundary
    evaluate coboundary(cochain) on face
    evaluate cochain on boundary(face)
    require the two exact integers to agree and be nonzero

    analyze homology in degrees one and two together
    record free ranks and torsion orders
    select two closed degree-one cycle representatives
    select one closed fundamental degree-two representative

    construct two closed integral one-cochains
    evaluate them on both selected one-cycles
    evaluate both cup orders and both squares on the fundamental cycle
    require unit determinants and graded-antisymmetric fundamental pairings

    project only the two selected cycles to binary64 display data
    render two cycle figures; make no invariant claim about their ordering
    ~~~
    """)
    return


@app.cell
def _(analyze_integral_homology, torus):
    domain, geometry = torus(12, 8)
    chains = domain.chain_complex()
    cochains = chains.dual()
    zero = ((), ())

    face = chains[2].element({0: 1})
    face_boundary = chains.boundary(2).apply(face)
    boundary_squared = chains.boundary(1).apply(face_boundary).to_python_copy()
    edge_cochain = cochains[1].element({0: 1})
    stokes_pair = (
        cochains.coboundary(1).apply(edge_cochain).evaluate(face),
        edge_cochain.evaluate(face_boundary),
    )

    homology = analyze_integral_homology(chains, [1, 2])
    group = homology[1]
    surface_group = homology[2]
    primal_cycles = tuple(group.free_cycle(index) for index in range(group.free_rank))
    fundamental_cycle = surface_group.free_cycle(0)
    primal_boundaries = tuple(
        chains.boundary(1).apply(cycle).to_python_copy() for cycle in primal_cycles
    )
    fundamental_boundary = chains.boundary(2).apply(fundamental_cycle).to_python_copy()

    dual_basis = domain.dual_cycles()
    cocycles = tuple(dual_basis.cocycle(index) for index in range(dual_basis.rank))
    dual_coboundaries = tuple(
        cochains.coboundary(1).apply(cocycle).to_python_copy() for cocycle in cocycles
    )
    period_matrix = tuple(
        tuple(cocycle.evaluate(cycle) for cycle in primal_cycles)
        for cocycle in cocycles
    )
    period_determinant = (
        period_matrix[0][0] * period_matrix[1][1]
        - period_matrix[0][1] * period_matrix[1][0]
    )
    cup_matrix = tuple(
        tuple(left.cup(right).evaluate(fundamental_cycle) for right in cocycles)
        for left in cocycles
    )
    cup_determinant = (
        cup_matrix[0][0] * cup_matrix[1][1] - cup_matrix[0][1] * cup_matrix[1][0]
    )

    if (
        boundary_squared != zero
        or stokes_pair[0] != stokes_pair[1]
        or stokes_pair[0] == 0
    ):
        raise RuntimeError("exact boundary or Stokes law failed")
    if (
        (group.free_rank, group.torsion_orders) != (2, ())
        or (surface_group.free_rank, surface_group.torsion_orders) != (1, ())
        or primal_boundaries != (zero, zero)
        or fundamental_boundary != zero
    ):
        raise RuntimeError("torus homology certificate failed")
    if (
        dual_basis.rank != 2
        or dual_coboundaries != (zero, zero)
        or abs(period_determinant) != 1
    ):
        raise RuntimeError("integral dual-cycle certificate failed")
    if (
        cup_matrix[0][0] != 0
        or cup_matrix[1][1] != 0
        or cup_matrix[0][1] != -cup_matrix[1][0]
        or abs(cup_matrix[0][1]) != 1
        or cup_determinant != 1
    ):
        raise RuntimeError("torus cup-pairing certificate failed")

    homology_evidence = {
        "simplex_counts": tuple(domain.simplex_count(k) for k in range(3)),
        "boundary_squared": boundary_squared,
        "stokes_pair": stokes_pair,
        "h1_structure": (group.free_rank, group.torsion_orders),
        "h2_structure": (surface_group.free_rank, surface_group.torsion_orders),
        "cycle_term_counts": tuple(
            len(cycle.to_python_copy()[0]) for cycle in primal_cycles
        ),
        "period_matrix": period_matrix,
        "period_determinant": period_determinant,
        "cup_matrix": cup_matrix,
        "cup_determinant": cup_determinant,
    }
    return geometry, group, homology_evidence


@app.cell
def _(homology_evidence, mo):
    mo.md(rf"""
    ## 6. Exact evidence

    Every entry below is an integer equality or quotient invariant; no numerical
    tolerance is involved.

    | Certificate | Exact result | Meaning |
    |:--|:--|:--|
    | Vertices, edges, faces | {homology_evidence["simplex_counts"]} | Finite torus chain groups |
    | Boundary of a face boundary | {homology_evidence["boundary_squared"]} | \(\partial_1\partial_2=0\) on the selected face |
    | Two Stokes routes | {homology_evidence["stokes_pair"]} | Equal and nonzero, so the sign is exercised |
    | \(H_1\): free rank, torsion | {homology_evidence["h1_structure"]} | \(H_1\cong\mathbb Z^2\) |
    | \(H_2\): free rank, torsion | {homology_evidence["h2_structure"]} | \(H_2\cong\mathbb Z\) |
    | Selected \(H_1\) cycle term counts | {homology_evidence["cycle_term_counts"]} | Representative data, not invariants |
    | Dual/primal period matrix | {homology_evidence["period_matrix"]} | Determinant \(={homology_evidence["period_determinant"]}\), a unit |
    | Fundamental cup matrix | {homology_evidence["cup_matrix"]} | Antisymmetric with determinant \(={homology_evidence["cup_determinant"]}\) |

    The quotient rows establish the two homology groups. The period matrix checks
    that the selected cocycles detect both selected cycle classes, while the cup
    matrix supplies the cohomological intersection witness. These roles are
    related but not interchangeable.
    """)
    return


@app.cell
def _(geometry, group, mo, plot_homology_cycle):
    cycle_figures = [
        plot_homology_cycle(
            geometry, group, index, title=f"Selected H1 representative {index + 1}"
        )
        for index in range(group.free_rank)
    ]
    mo.vstack(
        [
            mo.md(r"""
            ## 7. Selected representatives

            Each plot is a binary64 display projection of one exact sparse integer
            cycle. Geometry, color, and line width aid recognition only; they do
            not alter the quotient class or certify that this ordering survives a
            relabeling.
            """),
            mo.hstack(cycle_figures),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation and limits

    The chain-complex identity first makes boundaries into cycles; the quotient
    then forgets which closed representatives differ by filled chains. Algebraic
    duality evaluates those classes without a metric, and the cup product combines
    two degree-one detectors into an oriented degree-two pairing.

    For this torus, exact computation establishes free ranks two and one in
    degrees one and two, no torsion in either degree, and a unimodular
    antisymmetric pairing on the chosen fundamental class. The pictures show two
    useful representatives, not canonical longitude and meridian names. Nothing
    here asserts invariance of representative sparsity, coefficient order, or
    plotted appearance under remeshing or arbitrary relabeling.
    """)
    return


if __name__ == "__main__":
    app.run()
