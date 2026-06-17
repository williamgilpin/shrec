"""Benchmark B1 (plan 0009) — accuracy across (Connectivity × Reconstructor).

Scores stage combinations the 0008 refactor enabled against a known driver, to
answer "which combinations are worth using?" and confirm the canonical presets
are the right defaults. Writes a CSV next to the other benchmark scores.

Run:  uv run python benchmarks/pipeline_combo_sweep.py
Out:  benchmarks/pipeline_combo_scores.csv

Not part of the pytest path — this is a sweep (minutes), and doubles as a
regression baseline. The capability assertions in tests/test_pipeline.py
(MM41) are the fast guard; this is the quantitative picture.

Sensible-combo note (see 0009 / MM41): Fiedler & Leiden consume an affinity
(any connectivity); UnionFind needs a *sparse/binary* graph, so it pairs with
ExpKernel(sparsify=True); Isomap wants a dissimilarity. Simplicial+UnionFind is
included to *show* the degeneracy (a dense affinity collapses to one component).
"""
import csv
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from shrec.models import ShrecPipeline
from shrec.recurrence.connectivity import (
    CommonNeighborsConnectivity,
    ExpKernelConnectivity,
    SimplicialConnectivity,
)
from shrec.reconstruct import (
    FiedlerReconstructor,
    IsomapReconstructor,
    LeidenReconstructor,
    UnionFindReconstructor,
)

N_SWEEP = (4, 8, 16)
SEEDS = (0, 1, 2)
T = 300


# --- synthetic driver-response systems --------------------------------------

def discrete_dataset(n_responses, seed):
    """Period-2 driver weakly coupled into chaotic logistic responses."""
    rng = np.random.default_rng(seed)
    z = np.where(np.arange(T) % 2 == 0, 0.0, 1.0)
    r = rng.uniform(3.81, 3.97, size=n_responses)
    X = np.empty((T, n_responses))
    for k in range(n_responses):
        x = np.empty(T)
        x[0] = rng.uniform(0.1, 0.9)
        for t in range(T - 1):
            x[t + 1] = np.clip(r[k] * x[t] * (1 - x[t]) + 0.5 * z[t], 0.0, 1.0)
        X[:, k] = x
    return X, z


def continuous_dataset(n_responses, seed):
    """Smooth sinusoidal driver into chaotic logistic responses, light noise."""
    rng = np.random.default_rng(seed)
    z = 0.5 + 0.4 * np.sin(2 * np.pi * np.arange(T) / 90.0)
    r = rng.uniform(3.7, 3.9, size=n_responses)
    X = np.empty((T, n_responses))
    for k in range(n_responses):
        x = np.empty(T)
        x[0] = rng.uniform(0.1, 0.9)
        for t in range(T - 1):
            x[t + 1] = np.clip(r[k] * x[t] * (1 - x[t]) + 0.4 * z[t], 0.0, 1.0)
        X[:, k] = x + 0.03 * rng.standard_normal(T)
    return X, z


# --- the combo grid ----------------------------------------------------------
# (label, connectivity factory, reconstructor factory, task, preset?)
COMBOS = [
    ("Simplicial+Leiden",   lambda: SimplicialConnectivity(),
     lambda: LeidenReconstructor(random_state=1), "discrete", "RecurrenceClustering"),
    ("Simplicial+Fiedler",  lambda: SimplicialConnectivity(),
     lambda: FiedlerReconstructor(), "continuous", "RecurrenceManifold"),
    ("ExpKernel+Leiden",    lambda: ExpKernelConnectivity(sparsify=False),
     lambda: LeidenReconstructor(random_state=1), "discrete", ""),
    ("ExpKernel+Fiedler",   lambda: ExpKernelConnectivity(sparsify=False),
     lambda: FiedlerReconstructor(), "continuous", ""),
    ("ExpKernel+UnionFind", lambda: ExpKernelConnectivity(sparsify=True),
     lambda: UnionFindReconstructor(), "discrete", "ClassicalRecurrenceClustering"),
    ("CommonNbr+Fiedler",   lambda: CommonNeighborsConnectivity(),
     lambda: FiedlerReconstructor(), "continuous", ""),
    ("CommonNbr+Isomap",    lambda: CommonNeighborsConnectivity(),
     lambda: IsomapReconstructor(n_components=1), "continuous", "HirataNomuraIsomap"),
    ("Simplicial+UnionFind", lambda: SimplicialConnectivity(),
     lambda: UnionFindReconstructor(), "discrete", ""),  # expected degenerate
]


def score(labels, z, task):
    labels = np.asarray(labels).ravel()
    if task == "discrete":
        return adjusted_rand_score(labels, z.astype(int))
    return abs(spearmanr(labels, z).correlation)


def run():
    rows = []
    for label, conn_f, rec_f, task, preset in COMBOS:
        for n in N_SWEEP:
            vals = []
            for seed in SEEDS:
                X, z = (discrete_dataset if task == "discrete"
                        else continuous_dataset)(n, seed)
                try:
                    out = ShrecPipeline(
                        connectivity=conn_f(), reconstructor=rec_f(),
                    ).fit(X).labels_
                    vals.append(score(out, z, task))
                except Exception as e:  # noqa: BLE001 — record, don't abort the sweep
                    vals.append(np.nan)
                    print(f"  {label} N={n} seed={seed}: {type(e).__name__}: {e}")
            mean = float(np.nanmean(vals))
            rows.append({
                "combo": label, "preset": preset, "task": task,
                "metric": "ARI" if task == "discrete" else "|spearman|",
                "N": n, "score": round(mean, 4),
            })
            print(f"{label:22s} N={n:2d}  {rows[-1]['metric']:11s} {mean:.3f}"
                  + (f"   [{preset}]" if preset else ""))

    out_path = os.path.join(os.path.dirname(__file__), "pipeline_combo_scores.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["combo", "preset", "task", "metric", "N", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    run()
