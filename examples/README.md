# PolyGeo mathematical studies

These executable [Marimo](https://marimo.io/) notebooks develop one mathematical
question through a continuous statement, discrete derivation, mathematical
pseudocode, bounded computation, independent evidence, visualization where useful,
and an explicit claim boundary. The table is the suggested reading order.

| Group | Study | Learning outcome | Prior study used |
|---|---|---|---|
| Surface | [`surface/curvature.py`](surface/curvature.py) | Integrated Gaussian curvature on a sphere and torus | None |
| PDE | [`pde/poisson.py`](pde/poisson.py) | Compatible mean-zero Poisson solution | Curvature's measure/density distinction |
| Surface | [`surface/conformal_map.py`](surface/conformal_map.py) | Anchored conformal least squares with rank, residual, and orientation evidence | None |
| PDE | [`pde/heat_distance.py`](pde/heat_distance.py) | Heat-method distance with analytic sphere evidence | Poisson |
| Topology | [`topology/homology.py`](topology/homology.py) | Exact torus homology, Stokes pairing, and cup intersection | None |
| Topology | [`topology/hodge_decomposition.py`](topology/hodge_decomposition.py) | Reconstruction from exact, coexact, and harmonic parts | Poisson and homology |
| Fields | [`fields/holonomy.py`](fields/holonomy.py) | Local curvature versus global holonomy | Homology |
| Fields | [`fields/boundary_direction.py`](fields/boundary_direction.py) | Boundary alignment without arbitrary branch loss | Holonomy |

Exact evidence means an integer, rational, or combinatorial identity. Algebraic
evidence checks a finite operator or solve; geometric evidence compares a computed
quantity with a geometric law or analytic reference. A figure communicates an
already stated quantity and is not used as proof by itself.

Launch a study from the repository root:

```bash
uv run --extra mesh --extra plot marimo edit examples/surface/curvature.py
```

`examples/support/meshes.py` owns the deterministic meshes shared by the studies.
The canonical check and export matrix lives in the verification workflow.
