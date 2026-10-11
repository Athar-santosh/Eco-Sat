# vegsens — label-free vegetation-sensitivity mapping and corridor screening

Code, synthetic data generator and experiment logs for the paper

> Meenakshi and Rajesh T. M., *Label-Free Vegetation-Sensitivity Mapping for Eco-Aware Corridor Screening: A Superpixel–Mixture Pipeline Coupled with Standard Path-Search Solvers* (Dayananda Sagar University).

Every number, table and figure in Section 5 and Appendix B of the paper is produced by the two scripts in `experiments/`; nothing is entered by hand.

## Contents

| Path | Purpose |
|---|---|
| `vegsens/scenes.py` | Procedural generator for the five synthetic 220 × 220 R/G/B/NIR scenes S1–S5 with construction labels (seeds 101–505) |
| `vegsens/indices.py` | NDVI, VARI, ExG, GLI, robust min–max normalisation, ESS (Eq. 8), cost surface (Eq. 14) |
| `vegsens/segmentation.py` | SLIC + 8-component GMM + threshold-free class rules (Algorithm 4) + MRF-lite; K-means / per-pixel GMM baselines; χ, OA, κ |
| `vegsens/graphsearch.py` | numba A* / Dijkstra on the 8-connected grid (Eq. 16 edge weights) |
| `vegsens/solvers.py` | A*, Dijkstra, eco-aware ACO, NSGA-II, A*-seeded memetic NSGA-II |
| `vegsens/metrics.py` | f_len, f_eco, A_eco, route metrics, 2-D hypervolume |
| `experiments/run_all.py` | Runs all experiments; writes `results/*.pkl`, `results/route_runs.csv`, `results/segmentation.csv` |
| `experiments/analyze.py` | Builds `results/summary.json` and all figures in `results/figures/` |
| `app.py` | Streamlit screening application |
| `tests/` | Unit tests (A* = Dijkstra cost, crossover connectivity, loop removal, hypervolume) |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python experiments/run_all.py --runs 40      # ~30 min on one CPU core
python experiments/analyze.py                # tables (summary.json) and figures
pytest -q                                    # unit tests
streamlit run app.py                         # interactive application
```

Single stages: `--stage seg|sens|eff|route`. The logs used for the paper are included in `results/`, so `analyze.py` can be run without re-running the experiments.

## Key results (representative scene S3)

| Solver | Length (px) | Mean ESS | Route-accumulated ESS |
|---|---|---|---|
| A* | 292.1 | 0.269 | 79.5 |
| Dijkstra | 292.1 | 0.269 | 79.5 |
| ACO | 774.7 | 0.447 | 341.3 |
| NSGA-II | 288.3 | 0.428 | 123.4 |
| Memetic | 283.3 | 0.318 | 92.6 |

Segmentation OA 0.899–0.966 over S1–S5. Stochastic solvers: median of 40 runs; NSGA-II and memetic: knee of the final front.

## Important scope notes

* All evaluation scenes are **synthetic**; accuracies measure agreement with the generator, and segmenter settings were fixed on these same scenes.
* ESS is a **vegetation-greenness** score; water and wetlands receive low cost unless a protected-area layer is added.
* Routes are 8-connected pixel chains without geometric design constraints: a screening tool, not a highway-alignment optimiser.

## Environment used for the paper

Python 3.12, NumPy 2.4, SciPy 1.17, scikit-learn 1.8, scikit-image 0.26, numba 0.68, single CPU core (Linux).

## License

MIT (see `LICENSE`). If you use this code, please cite the paper.
