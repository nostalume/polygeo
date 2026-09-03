# PolyGeo mathematical studies

These executable [Marimo](https://marimo.io/) notebooks connect one mathematical
question to the direct PolyGeo API, a real computation, a visualization where
useful, and an independently calculated check.

| Group | Study | Learning outcome |
|---|---|---|
| Surface | [`surface/curvature.py`](surface/curvature.py) | Gaussian curvature on a sphere and torus |
| PDE | [`pde/poisson.py`](pde/poisson.py) | Compatible mean-zero Poisson solution |
| Surface | [`surface/conformal_map.py`](surface/conformal_map.py) | Two-anchor LSCM with rank, residual, and orientation evidence |
| PDE | [`pde/heat_distance.py`](pde/heat_distance.py) | Heat-method distance with analytic sphere evidence |
| Topology | [`topology/homology.py`](topology/homology.py) | Exact torus homology, Stokes pairing, and cup intersection |
| Topology | [`topology/hodge_decomposition.py`](topology/hodge_decomposition.py) | Reconstruction from exact, coexact, and harmonic parts |
| Fields | [`fields/holonomy.py`](fields/holonomy.py) | Local versus global holonomy obstruction |
| Fields | [`fields/boundary_direction.py`](fields/boundary_direction.py) | Order-four boundary alignment, branch choice, and exact singularity evidence |

Launch a study from the repository root:

```bash
uv run --extra mesh --extra plot marimo edit examples/surface/curvature.py
```

`examples/support/meshes.py` owns the deterministic meshes shared by the studies.
The canonical check and export matrix lives in the verification workflow.
