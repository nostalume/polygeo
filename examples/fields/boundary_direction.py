import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from polygeo.geometry import Geometry, TriangleSurface
    from polygeo.plot import direction as plot_direction, vectors as plot_vectors
    from examples.support.meshes import disk

    return Geometry, TriangleSurface, disk, mo, np, plot_direction, plot_vectors


@app.cell
def _(mo):
    mo.md(r"""
    # Boundary-aligned symmetric direction without branch loss

    ## Question and prerequisites

    How can an order-four direction align to an oriented disk boundary without
    arbitrarily choosing one of its four equivalent rays, and how do its interior
    singularities balance the boundary winding?

    The inputs are a connected oriented triangle disk, a nondegenerate realization
    in three-dimensional Euclidean space, a positive discrete metric, and symmetry
    order \(N=4\). Face frames and Levi--Civita transport are supplied by the
    preceding connection study. The goal is one quotient-valued field, integer
    interior charges, oriented boundary turns, and a disk balance certificate.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Quotienting removes a false branch choice

    A unit tangent vector has a face-frame coordinate \(u=e^{i\theta}\). An
    order-\(N\) direction regards the \(N\) rotations

    \[
    u_k=e^{i(\theta+2\pi k/N)},\qquad k=0,\ldots,N-1,
    \]

    as the same object. Thus \(\theta\sim\theta+2\pi k/N\). Storing one angle
    representative would introduce information that the direction does not own.
    Instead take the invariant power

    \[
    P_N([u])=z_N=u^N=e^{iN\theta}.
    \]

    This map is well defined because

    \[
    \left(u e^{2\pi i k/N}\right)^N=u^N e^{2\pi i k}=u^N.
    \]

    It also loses exactly the intended distinction: if \(u^N=v^N\), then
    \((v/u)^N=1\), so \(v/u=e^{2\pi i k/N}\) and \(u\sim v\). The power coordinate
    therefore represents one full equivalence class, not one privileged ray.
    Its display branches are recovered only at the visualization boundary:

    \[
    u_k=\exp\!\left(i\frac{\operatorname{Arg}z_N+2\pi k}{N}\right).
    \]
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Transported power mismatch generates integer charge

    Let adjacent oriented faces \(f,g\) have unit-complex Levi--Civita transport
    \(q_{fg}=e^{i\omega_{fg}}\). Transport acts on a vector coordinate by
    multiplication with \(q_{fg}\); hence on the quotient coordinate it acts by

    \[
    Q_{fg}=q_{fg}^{N}=e^{iN\omega_{fg}}.
    \]

    For face powers \(z_f,z_g\), compare the observed target with the transported
    source in their common target frame:

    \[
    m_{fg}=z_g\,\overline{Q_{fg}z_f},\qquad
    \mu_{fg}=W(\operatorname{Arg}m_{fg}),\qquad
    W(t)=\operatorname{atan2}(\sin t,\cos t).
    \]

    The field is transported exactly when \(m_{fg}=1\), equivalently
    \(\mu_{fg}=0\). Around an interior vertex \(v\), let \([v:e]\) be the canonical
    vertex--edge incidence and let \(s_{f(e),e}\) be the signed occurrence of edge
    \(e\) in the lower-index adjacent face. Their product fixes the traversal sign
    without choosing a drawing. If \(K_v\) is the integrated angle defect, the
    unwrapped power-angle total is

    \[
    A_v=N K_v+\sum_{e\ni v}[v:e]s_{f(e),e}\,\mu_e,
    \qquad \widehat q_v=\frac{A_v}{2\pi}.
    \]

    Each term lives in the same oriented power-angle space. Telescoping cancels
    face phases; the angle defect restores the frame rotation accumulated around
    the loop. Therefore a representable field requires

    \[
    q_v=\operatorname{round}(\widehat q_v)\in\mathbb Z,
    \qquad r_v=|\widehat q_v-q_v|\ll 1.
    \]

    If \(q_v\ne0\), a lifted ray winds by \(2\pi q_v/N\). No continuous,
    nonsingular single-valued branch can extend across \(v\); selecting branch zero
    merely places a cut and does not remove the obstruction.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Oriented boundary alignment and the relative index law

    The coherent face orientation induces each boundary edge direction \(a\to b\).
    Its unit tangent and face-frame coordinate are

    \[
    t_e=\frac{p_b-p_a}{\lVert p_b-p_a\rVert},\qquad
    \tau_e=\langle t_e,u_f\rangle+i\langle t_e,v_f\rangle.
    \]

    Zero-offset tangential alignment requests \(z_f=\tau_e^N\). If several
    boundary edges meet one face, their length-weighted power directions are added
    and normalized before imposing the request; this averaging remains in quotient
    space and never averages arbitrary roots.

    Along an oriented boundary component \(C=(e_0,\ldots,e_{m-1})\), measure the
    field relative to its tangent by \(\rho_e=z_f\overline{\tau_e^N}\). Consecutive
    residuals use the same principal wrap as interior crossings:

    \[
    \delta_j=W\!\left(\operatorname{Arg}
      (\rho_{e_{j+1}}\overline{\rho_{e_j}})\right),\qquad
    b_C=-\frac{1}{2\pi}\sum_{j=0}^{m-1}\delta_j.
    \]

    The minus sign records that the boundary frame, rather than the field, is the
    moving reference. Interior and boundary windings then share one orientation
    convention and satisfy the relative index law

    \[
    \sum_{v\in\operatorname{int}M}q_v+\sum_C b_C=N\chi(M),
    \qquad \chi(M)=V-E+F.
    \]

    Our disk has \(\chi=1\), so an order-four field must contribute exactly four.
    The numerical construction minimizes weighted connection deviation while the
    boundary powers are fixed, first continuously and then inside one admitted
    integer lift sector. It does not compare every sector and makes no global-
    minimum claim across them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Mathematical pseudocode

    ~~~text
    construct one oriented disk in three dimensions and choose N = 4
    encode zero-offset boundary tangents in the Nth-power quotient
    solve once inside one admitted integer lift sector
    compare adjacent face powers after Nth-power parallel transport
    for every interior vertex:
        accumulate consistently oriented wrapped power mismatches and angle defect
        divide the unwrapped total by 2 pi and certify the nearest integer charge
    around the oriented boundary:
        compare each face power with its tangent power
        wrap consecutive relative-angle increments and recover the boundary turn
    check charge sum + boundary-turn sum = N (V - E + F)
    reconstruct branch zero once and verify that its Nth power returns the class
    render all N rays, then render branch zero as a lossy comparison
    ~~~
    """)
    return


@app.cell
def _(Geometry, TriangleSurface, disk, np):
    order = 4
    domain, planar = disk(3, 16)
    _xy = planar.positions_numpy_copy()
    _height = 0.16 * (1.0 - (_xy[:, 0] / 1.4) ** 2 - (_xy[:, 1] / 0.8) ** 2)
    geometry = Geometry.from_positions(domain, np.column_stack((_xy, _height)))
    surface = TriangleSurface.admit(geometry)
    field = surface.boundary_direction(order, geometry.metric(), 0.0)
    singularities = field.singularities()
    return domain, field, geometry, order, singularities, surface


@app.cell
def _(domain, field, geometry, np, order, singularities, surface):
    _tau = 2.0 * np.pi

    def _complex_rows(rows):
        return rows[:, 0] + 1j * rows[:, 1]

    def _wrap(values):
        return np.arctan2(np.sin(values), np.cos(values))

    _power = _complex_rows(field.power_directions_numpy_copy())
    _d2_signs, _d2_faces, _d2_starts, _ = domain.boundary_parts_numpy_copy(2)
    _connection = field.connection.connection
    _interior_edges = _connection.interior_edge_indices_numpy_copy()
    _starts = _d2_starts[_interior_edges]
    _left, _right = _d2_faces[_starts], _d2_faces[_starts + 1]
    _left_is_source = _left < _right
    _source = np.where(_left_is_source, _left, _right)
    _target = np.where(_left_is_source, _right, _left)
    _source_sign = np.where(_left_is_source, _d2_signs[_starts], _d2_signs[_starts + 1])
    _transport = _complex_rows(_connection.transports_numpy_copy())
    _crossings = _wrap(
        np.angle(_power[_target] * np.conj(_transport * _power[_source]))
    )

    _levi_civita = surface.levi_civita()
    _levi_power = _complex_rows(_levi_civita.transports_numpy_copy()) ** order
    _mismatches = _wrap(
        np.angle(_power[_target] * np.conj(_levi_power * _power[_source]))
    )
    _edge_count = domain.simplex_count(1)
    _signed_mismatch = np.zeros(_edge_count)
    _signed_mismatch[_interior_edges] = _source_sign * _mismatches
    _curvature = surface.gaussian_curvature_measure().coefficients_numpy_copy()
    _raw_charges = (
        order * _curvature + domain.boundary_scipy_copy(1) @ _signed_mismatch
    ) / _tau
    _interior = ~domain.boundary_mask_numpy_copy(0)
    _rounded_charges = np.rint(_raw_charges).astype(np.int64)
    _nonzero = _interior & (_rounded_charges != 0)
    _charge_indices = tuple(int(value) for value in np.flatnonzero(_nonzero))
    _charges = tuple(int(value) for value in _rounded_charges[_nonzero])
    _charge_residual = float(
        np.max(np.abs(_raw_charges[_interior] - _rounded_charges[_interior]))
    )

    _edge_rows = domain.simplices_numpy_copy(1)
    _boundary_edges = np.flatnonzero(np.diff(_d2_starts) == 1)
    _rows = _d2_starts[_boundary_edges]
    _boundary_faces = _d2_faces[_rows]
    _boundary_signs = _d2_signs[_rows]
    _endpoints = _edge_rows[_boundary_edges]
    _a = np.where(_boundary_signs == 1, _endpoints[:, 0], _endpoints[:, 1])
    _b = np.where(_boundary_signs == 1, _endpoints[:, 1], _endpoints[:, 0])
    _points = geometry.positions_numpy_copy()
    _displacements = _points[_b] - _points[_a]
    _lengths = np.linalg.norm(_displacements, axis=1)
    _tangents = _displacements / _lengths[:, None]
    _first = surface.first_frame_axes_numpy_copy()
    _second = surface.second_frame_axes_numpy_copy()
    _local_tangents = np.einsum(
        "ij,ij->i", _tangents, _first[_boundary_faces]
    ) + 1j * np.einsum("ij,ij->i", _tangents, _second[_boundary_faces])
    _tangent_power = _local_tangents**order
    _tangent_power /= np.abs(_tangent_power)
    if len(np.unique(_boundary_faces)) != len(_boundary_faces):
        raise RuntimeError("the disk probe requires one boundary edge per face")
    _boundary_alignment_error = float(
        np.max(np.abs(_power[_boundary_faces] - _tangent_power))
    )

    _residual_by_edge = np.zeros(_edge_count, dtype=np.complex128)
    _residual_by_edge[_boundary_edges] = _power[_boundary_faces] * np.conj(
        _tangent_power
    )
    _cycle = domain.disk_boundary_vertices_numpy_copy()
    _edge_at_source = dict(zip(_a, _boundary_edges, strict=True))
    _ordered_edges = np.asarray([_edge_at_source[vertex] for vertex in _cycle])
    _relative = _residual_by_edge[_ordered_edges]
    _increments = _wrap(np.angle(np.roll(_relative, -1) * np.conj(_relative)))
    _raw_boundary_turn = -float(np.sum(_increments)) / _tau
    _boundary_turn = int(round(_raw_boundary_turn))
    _boundary_residual = abs(_raw_boundary_turn - _boundary_turn)

    _native_indices, _native_charges = singularities.charges.to_python_copy()
    _chi = sum((-1) ** degree * domain.simplex_count(degree) for degree in range(3))
    _balance = (sum(_charges) + _boundary_turn, order * _chi)
    branch_zero = field.ambient_branch_numpy_copy(0)
    _branch_vectors = branch_zero.values_numpy_copy()
    _local_branch = np.einsum("ij,ij->i", _branch_vectors, _first) + 1j * np.einsum(
        "ij,ij->i", _branch_vectors, _second
    )
    _branch_power_error = float(np.max(np.abs(_local_branch**order - _power)))
    _crossing_error = float(np.max(np.abs(_crossings)))
    _quantization_error = max(_charge_residual, _boundary_residual)
    _limit = singularities.residual_limit
    _certificate_errors = (
        _boundary_alignment_error,
        _branch_power_error,
        abs(_crossing_error - field.connection.crossing_error),
        abs(_quantization_error - singularities.maximum_quantization_residual),
    )
    if (
        _charge_indices != tuple(_native_indices)
        or _charges != tuple(_native_charges)
        or (_boundary_turn,) != singularities.boundary_turns_python_copy()
        or _balance[0] != _balance[1]
        or max(*_certificate_errors, _crossing_error, _quantization_error) > _limit
    ):
        raise RuntimeError("boundary-direction certificate failed")

    direction_evidence = {
        "simplex_counts": tuple(domain.simplex_count(k) for k in range(3)),
        "charge_indices": _charge_indices,
        "charges": _charges,
        "boundary_turns": (_boundary_turn,),
        "euler_characteristic": _chi,
        "balance": _balance,
        "crossing_error": _crossing_error,
        "boundary_alignment_error": _boundary_alignment_error,
        "charge_quantization_residual": _charge_residual,
        "boundary_quantization_residual": _boundary_residual,
        "branch_zero_power_error": _branch_power_error,
        "residual_limit": _limit,
    }
    return branch_zero, direction_evidence


@app.cell
def _(direction_evidence, mo):
    _charges = direction_evidence["charges"]
    _indices = direction_evidence["charge_indices"]
    _turns = direction_evidence["boundary_turns"]
    _balance = direction_evidence["balance"]
    mo.md(rf"""
    ## 5. Independent numerical evidence

    The lifted disk has simplex counts {direction_evidence["simplex_counts"]} and
    Euler characteristic {direction_evidence["euler_characteristic"]}.

    | Exact certificate | Observed value |
    |:--|:--|
    | Nonzero interior vertices | {_indices} |
    | Integer charges | {_charges} |
    | Oriented boundary turns | {_turns} |
    | Charge + boundary versus \(N\chi\) | {_balance[0]} = {_balance[1]} |

    | Algebraic or geometric certificate | Error |
    |:--|--:|
    | Transported power crossing | {direction_evidence["crossing_error"]:.3e} |
    | Zero-offset boundary alignment | {direction_evidence["boundary_alignment_error"]:.3e} |
    | Interior-charge quantization | {direction_evidence["charge_quantization_residual"]:.3e} |
    | Boundary-turn quantization | {direction_evidence["boundary_quantization_residual"]:.3e} |
    | Branch-zero power reconstruction | {direction_evidence["branch_zero_power_error"]:.3e} |

    The floating-point witnesses are below the declared limit
    {direction_evidence["residual_limit"]:.3e}. The integer balance is exact after
    the independently computed winding values pass their quantization tests.
    """)
    return


@app.cell
def _(
    branch_zero,
    field,
    mo,
    plot_direction,
    plot_vectors,
):
    _class_figure = plot_direction(
        field, scale=0.16, title="One order-four class: all equivalent rays"
    )
    _branch_figure = plot_vectors(
        branch_zero, scale=0.16, title="Branch zero only: an information-losing lift"
    )
    mo.vstack(
        [
            mo.md(r"""
            ## 6. Quotient field versus one branch

            In the first figure, four blue rays at a face encode one equivalence
            class; none is an extra solution. The second figure draws only branch
            zero in amber. Its line body and arrowhead share one visibility group,
            so the legend cannot hide one while leaving the other behind.
            """),
            _class_figure,
            _branch_figure,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Interpretation and limits

    Power coordinates remove a representational ambiguity before solving: the
    stored object is the symmetric class itself. Transport mismatch and angle
    defect then generate quantized interior charges, while comparison with the
    induced boundary tangent generates the relative boundary contribution. Their
    exact disk balance is topological; the small floating-point residuals certify
    that this particular computation represents those integers unambiguously.

    Branch zero is useful only as a local visual lift. It discards the other three
    equivalent rays and its cut cannot erase nonzero winding. The experiment uses
    one mesh, one embedding, order four, zero tangential offset, and one admitted
    integer sector. It does not establish convergence under refinement, uniqueness,
    or a global minimum over all lift sectors, and it makes no claim for
    nonorientable, degenerate, disconnected, or higher-genus boundary geometries.
    """)
    return


if __name__ == "__main__":
    app.run()
