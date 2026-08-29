# PolyGeo mathematical studies

These executable [Marimo](https://marimo.io/) notebooks connect one mathematical
question to the direct PolyGeo API, a real computation, a visualization where
useful, and an independently calculated check.

| Study | Learning outcome |
|---|---|
| `curvature.py` | Gaussian curvature on a sphere and torus |
| `poisson.py` | Compatible mean-zero Poisson solution |
| `harmonic_extension.py` | Boundary-preserving extension on an annulus |
| `mean_curvature_flow.py` | Atomic frozen-flow update |
| `homology.py` | Exact degree-one homology of a torus |
| `hodge_decomposition.py` | Reconstruction from exact, coexact, and harmonic parts |
| `connection_and_holonomy.py` | Local versus global holonomy obstruction |

Launch a study from the repository root:

```bash
uv run --extra mesh --extra plot marimo edit examples/curvature.py
```

`examples/support/meshes.py` owns the three deterministic meshes shared by the
studies. The canonical check and export matrix lives in the verification
workflow.
