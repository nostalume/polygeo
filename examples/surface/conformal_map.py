import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo.geometry import Geometry, TriangleSurface
    from polygeo.plot import geometry as plot_geometry
    from examples.support.meshes import disk

    return Geometry, TriangleSurface, disk, mo, np, plot_geometry


@app.cell
def _(mo):
    mo.md(r"""
    # Least-squares conformal map of a triangle disk

    ## Question and prerequisites

    How does a smooth conformality equation become a rectangular linear system for
    a piecewise-linear map, and why are two fixed boundary vertices enough to make
    its least-squares solution determinate?

    We need only oriented tangent planes, barycentric coordinates on a triangle,
    and linear least squares. The source is an oriented triangle disk in
    $\mathbb R^3$; the unknown target position of vertex $i$ is
    $q_i=(u_i,v_i)\in\mathbb R^2$.

    ## 1. Smooth conformality is a commutation equation

    Give both oriented tangent planes the quarter-turn operator

    $$
    J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
    $$

    An orientation-preserving differential is conformal exactly when it commutes
    with this complex structure:

    $$
    Df\,J=J\,Df.
    $$

    In oriented orthonormal source coordinates $(\xi,\eta)$, write

    $$
    Df=
    \begin{pmatrix}
    U_\xi&U_\eta\\
    V_\xi&V_\eta
    \end{pmatrix}.
    $$

    Multiplying out the two sides gives

    $$
    DfJ=
    \begin{pmatrix}U_\eta&-U_\xi\\V_\eta&-V_\xi\end{pmatrix},
    \qquad
    JDf=
    \begin{pmatrix}-V_\xi&-V_\eta\\U_\xi&U_\eta\end{pmatrix}.
    $$

    Equality is therefore the two real Cauchy--Riemann equations

    $$
    U_\xi-V_\eta=0,
    \qquad
    U_\eta+V_\xi=0.
    $$

    For a nonzero differential, these equations say that the two columns of $Df$
    are orthogonal, have equal length, and form the positive orientation. They are
    linear in the entries of $Df$, which is the key fact used below.

    ## 2. One triangle contributes two linear residuals

    Take one oriented source face $T=(i,j,k)$. Unfold its tangent plane isometrically
    into an oriented local frame, giving coordinates
    $p_\ell=(\xi_\ell,\eta_\ell)$. Let

    $$
    2A_T=
    \det\!\begin{pmatrix}
    \xi_j-\xi_i&\xi_k-\xi_i\\
    \eta_j-\eta_i&\eta_k-\eta_i
    \end{pmatrix}>0.
    $$

    The barycentric hat function $\lambda_i$ is affine, equals one at $p_i$, and
    vanishes on the edge $p_jp_k$. Solving those three affine constraints gives

    $$
    \nabla\lambda_i=
    \frac{1}{2A_T}
    \begin{pmatrix}
    \eta_j-\eta_k\\
    \xi_k-\xi_j
    \end{pmatrix},
    $$

    with the formulas for $\lambda_j$ and $\lambda_k$ obtained by cyclic
    permutation. Notice that these gradients sum to zero, as they must because
    $\lambda_i+\lambda_j+\lambda_k=1$.

    The piecewise-linear target map on this face is

    $$
    F_h=(U,V)=\sum_{\ell\in T}(u_\ell,v_\ell)\lambda_\ell.
    $$

    Hence its derivative is constant on $T$:

    $$
    \nabla U=\sum_{\ell\in T}u_\ell\nabla\lambda_\ell,
    \qquad
    \nabla V=\sum_{\ell\in T}v_\ell\nabla\lambda_\ell.
    $$

    Introduce $w_\ell=u_\ell+\mathrm i v_\ell$ and
    $c_\ell=\partial_\xi\lambda_\ell+
    \mathrm i\,\partial_\eta\lambda_\ell$. Substitution into the two
    Cauchy--Riemann equations packages them into one complex residual:

    $$
    r_T=\sum_{\ell\in T}c_\ell w_\ell,
    $$

    because

    $$
    \operatorname{Re}r_T=U_\xi-V_\eta,
    \qquad
    \operatorname{Im}r_T=U_\eta+V_\xi.
    $$

    More explicitly, if $c_\ell=a_\ell+\mathrm i b_\ell$, the columns belonging
    to $(u_\ell,v_\ell)$ are

    $$
    \begin{array}{c|cc}
      &u_\ell&v_\ell\\ \hline
    \operatorname{Re}r_T&a_\ell&-b_\ell\\
    \operatorname{Im}r_T&b_\ell&a_\ell
    \end{array}.
    $$

    Thus each face contributes exactly two real rows. A positive face-dependent
    scaling may balance the residuals without changing which maps satisfy
    $r_T=0$ exactly.

    ## 3. The global system and its four-dimensional freedom

    Stack the two rows from every face and order the unknowns as
    $x=(u_0,v_0,u_1,v_1,\ldots)^\mathsf T$. For $F$ faces and $V$ vertices,

    $$
    A x=0,
    \qquad A\in\mathbb R^{2F\times 2V}.
    $$

    Usually no noncollapsed piecewise-linear map satisfies every face equation
    exactly, so the discrete problem minimizes $\lVert Ax\rVert_2$. Adding either
    constant coordinate field leaves all derivatives unchanged, so $A$ has at
    least the two translation vectors in its kernel and is rank-deficient.

    Rotation and scale require a slightly different statement. Complex linearity
    gives

    $$
    \lVert A(e^{\mathrm i\theta}x)\rVert_2=\lVert Ax\rVert_2,
    \qquad
    \lVert A(sx)\rVert_2=|s|\lVert Ax\rVert_2.
    $$

    Rotation is therefore an equivariance of the residual norm, while scale
    homogeneity lets an unconstrained minimization collapse to a constant map at
    $s=0$. Together with translation they are the four real coordinate freedoms
    of a planar similarity, but rotation and scale are not generally two more
    literal kernel vectors for a curved source mesh.

    Choose two distinct, far-separated boundary vertices $a$ and $b$, and prescribe

    $$
    q_a=(0,0),\qquad q_b=(1,0).
    $$

    This removes all four similarity freedoms. Splitting the columns into free and
    fixed coordinates gives

    $$
    A_{\mathrm{free}}x_{\mathrm{free}}
    +A_{\mathrm{fixed}}x_{\mathrm{fixed}}=0,
    $$

    and therefore the single rectangular least-squares problem

    $$
    \underset{x_{\mathrm{free}}}{\operatorname{minimize}}
    \left\lVert A_{\mathrm{free}}x_{\mathrm{free}}-b\right\rVert_2,
    \qquad
    b=-A_{\mathrm{fixed}}x_{\mathrm{fixed}},
    $$

    where $A_{\mathrm{free}}\in\mathbb R^{2F\times 2(V-2)}$. No other boundary
    vertex is constrained.

    ## 4. What the diagnostics do—and do not—say

    The **expected rank** is the number of free scalar coordinates,
    $2(V-2)$. The **numerical rank** counts independent columns relative to a
    scale-aware floating-point threshold; equality with the expected rank says
    the anchored reduced system has no detected null direction.

    The dimensionless **normalized residual** is

    $$
    \rho=
    \frac{\lVert A_{\mathrm{free}}x_{\mathrm{free}}-b\rVert_2}
    {\lVert A_{\mathrm{free}}\rVert_F\lVert x_{\mathrm{free}}\rVert_2
    +\lVert b\rVert_2}.
    $$

    It measures how closely this discrete map satisfies the assembled
    Cauchy--Riemann equations at the problem's scale. A **condition indicator**,
    formed from the largest and smallest retained pivots of a rank-revealing
    factorization, instead warns how sensitively the solution may respond to
    perturbations. Small residual and good conditioning are different claims.

    Finally, for an oriented face with stored-orientation sign $\sigma_T$, define

    $$
    s_T=\sigma_T\det(q_j-q_i,\ q_k-q_i).
    $$

    This is twice its signed target area. The sign must include $\sigma_T$:
    simplex storage order need not itself agree with the disk orientation. The
    independent witness below uses raw target coordinates, so $s_T$ has units of
    target length squared. The certified witness normalizes each face by a local
    edge scale before applying a robust sign test; it is dimensionless. Their
    signs should agree, but their magnitudes should not be compared.

    If every $s_T>0$, each affine face is nondegenerate and locally preserves the
    chosen orientation. This does **not** prove that distant faces do not overlap,
    that the whole disk is globally injective, or that the piecewise-linear map is
    a smooth conformal map.

    ## 5. Pseudocode

    ```text
    construct an oriented triangle disk and lift its interior into R3
    list the B boundary vertices
    examine every unordered boundary pair and choose the largest separation
        # O(B^2), acceptable for this fixed small fixture
    fix the two selected target positions to (0, 0) and (1, 0)

    for each oriented face:                         # O(F) assembly
        unfold its tangent plane into an oriented orthonormal frame
        derive the three constant barycentric gradients
        form one complex Cauchy--Riemann residual
        append its real and imaginary parts as two matrix rows

    move the two fixed vertices' columns to the right-hand side
    solve the one rectangular least-squares problem for all free coordinates
    reconstruct every target vertex, including the exact anchors

    independently compute every oriented signed target area
    report expected rank, numerical rank, normalized residual,
        condition indicator, both minimum signed-area witnesses,
        and the exact-predicate fallback count
    render the source and target side by side; identify the anchors in prose
    ```
    """)
    return


@app.cell
def _(Geometry, TriangleSurface, disk, np):
    domain, planar = disk(3, 16)
    planar_positions = planar.positions_numpy_copy()
    height = 0.35 * (1.0 - np.sum(planar_positions * planar_positions, axis=1))
    source = Geometry.from_positions(
        domain, np.column_stack((planar_positions, height))
    )
    boundary = domain.disk_boundary_vertices_numpy_copy()
    boundary_positions = source.positions_numpy_copy()[boundary]
    separations = np.sum(
        (boundary_positions[:, None, :] - boundary_positions[None, :, :]) ** 2,
        axis=2,
    )
    separations[np.tril_indices_from(separations)] = -np.inf
    first_boundary_index, second_boundary_index = np.unravel_index(
        np.argmax(separations), separations.shape
    )
    anchors = (
        int(boundary[first_boundary_index]),
        int(boundary[second_boundary_index]),
    )
    solution = TriangleSurface.admit(source).conformal_map(anchors)
    mapped_geometry = solution.geometry
    mapped = mapped_geometry.positions_numpy_copy()

    faces = domain.simplices_numpy_copy(2)
    first_target_edge = mapped[faces[:, 1]] - mapped[faces[:, 0]]
    second_target_edge = mapped[faces[:, 2]] - mapped[faces[:, 0]]
    signed_twice_area = domain.orientations_numpy_copy(2) * (
        first_target_edge[:, 0] * second_target_edge[:, 1]
        - first_target_edge[:, 1] * second_target_edge[:, 0]
    )
    if not np.all(signed_twice_area > 0.0):
        raise RuntimeError("LSCM target contains a nonpositive oriented face")
    np.testing.assert_array_equal(mapped[list(anchors)], [[0.0, 0.0], [1.0, 0.0]])
    conformal_map_evidence = {
        "vertex_count": mapped.shape[0],
        "face_count": faces.shape[0],
        "boundary_vertex_count": boundary.shape[0],
        "row_count": 2 * faces.shape[0],
        "free_column_count": 2 * (mapped.shape[0] - 2),
        "anchors": anchors,
        "anchor_squared_separation": float(
            separations[first_boundary_index, second_boundary_index]
        ),
        "required_rank": solution.required_rank,
        "observed_rank": solution.observed_rank,
        "condition_indicator": solution.condition_indicator,
        "normalized_conformality_residual": solution.residual_bound,
        "minimum_native_normalized_signed_twice_area": (
            solution.minimum_normalized_signed_twice_area
        ),
        "minimum_independent_signed_twice_area": float(np.min(signed_twice_area)),
        "exact_fallback_faces": solution.exact_fallback_faces,
    }
    return conformal_map_evidence, mapped_geometry, source


@app.cell
def _(conformal_map_evidence, mo):
    mo.md(rf"""
    ## 6. Numerical evidence

    The fixture has {conformal_map_evidence["vertex_count"]} vertices,
    {conformal_map_evidence["face_count"]} faces, and
    {conformal_map_evidence["boundary_vertex_count"]} boundary vertices. Its
    reduced matrix is therefore
    ${conformal_map_evidence["row_count"]}\times
    {conformal_map_evidence["free_column_count"]}$.

    | Quantity | Observed value | Meaning |
    |:--|--:|:--|
    | Anchor vertex identities | {conformal_map_evidence["anchors"]} | Fixed to $(0,0)$ and $(1,0)$ |
    | Anchor squared source separation | {conformal_map_evidence["anchor_squared_separation"]:.6f} | Maximum among the unordered boundary pairs |
    | Numerical / expected rank | {conformal_map_evidence["observed_rank"]} / {conformal_map_evidence["required_rank"]} | Full reduced column rank detected |
    | Normalized conformality residual | {conformal_map_evidence["normalized_conformality_residual"]:.6e} | Small, but not zero, discrete CR mismatch |
    | Condition indicator | {conformal_map_evidence["condition_indicator"]:.6f} | Distinct from approximation error |
    | Minimum certified normalized signed twice-area | {conformal_map_evidence["minimum_native_normalized_signed_twice_area"]:.6e} | Dimensionless robust orientation witness |
    | Minimum independent raw signed twice-area | {conformal_map_evidence["minimum_independent_signed_twice_area"]:.6e} | Positive; units are target length squared |
    | Exact-predicate fallback faces | {conformal_map_evidence["exact_fallback_faces"]} | No ambiguous sign required exact resolution here |

    The two anchor positions were also checked for exact equality after
    reconstruction. Full rank removes the anchored linear nullspace; the residual
    quantifies the remaining least-squares mismatch; and the two positive area
    minima independently support local orientation preservation.
    """)
    return


@app.cell
def _(conformal_map_evidence, mo):
    mo.md(rf"""
    ## 7. Source and target

    The left view is the lifted disk in $\mathbb R^3$. The right view is the one
    planar least-squares realization. Boundary vertices
    {conformal_map_evidence["anchors"][0]} and
    {conformal_map_evidence["anchors"][1]} are the selected far-separated pair;
    they land exactly at $(0,0)$ and $(1,0)$, respectively. Every other boundary
    vertex remains an unknown determined by the same rectangular solve.
    """)
    return


@app.cell
def _(mapped_geometry, mo, plot_geometry, source):
    mo.hstack(
        [
            plot_geometry(source, title="Source disk in R3"),
            plot_geometry(mapped_geometry, title="LSCM target in R2"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Interpretation

    This experiment supports a precise, limited conclusion: after fixing planar
    similarity freedom, the reduced system has full detected column rank, its
    least-squares Cauchy--Riemann residual is small at the reported normalization,
    and every target face has positive oriented area by both a robust normalized
    witness and an independent raw determinant.

    It does not turn a finite triangle mesh into a smooth surface theorem. In
    particular, positive face areas are local statements and cannot rule out a
    global overlap between nonadjacent triangles. Global injectivity would require
    an additional argument or test.
    """)
    return


if __name__ == "__main__":
    app.run()
