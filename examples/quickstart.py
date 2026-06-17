"""SHREC quickstart — from a response ensemble to a recovered driver.

The minimal "does it work" example (plan 0009 / E1). Run:

    uv run python examples/quickstart.py

Writes `examples/quickstart.png`. Self-contained — generates a synthetic
driven-logistic ensemble so it needs no data files.
"""
import os

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shrec import RecurrenceManifold
from shrec.plotting import plot_driver_overlay, plot_recurrence_matrix

HERE = os.path.dirname(__file__)


def make_ensemble(T=400, n_responses=16, coupling=0.4, noise=0.03, seed=0):
    """A smooth hidden driver `z(t)` forced into `n_responses` chaotic logistic
    maps, each observed through light noise. Only `X` is given to SHREC; `z` is
    held back to score the reconstruction."""
    rng = np.random.default_rng(seed)
    z = 0.5 + 0.4 * np.sin(2 * np.pi * np.arange(T) / 90.0)
    r = rng.uniform(3.7, 3.9, size=n_responses)
    X = np.empty((T, n_responses))
    for k in range(n_responses):
        x = np.empty(T)
        x[0] = rng.uniform(0.1, 0.9)
        for t in range(T - 1):
            x[t + 1] = np.clip(r[k] * x[t] * (1 - x[t]) + coupling * z[t], 0.0, 1.0)
        X[:, k] = x + noise * rng.standard_normal(T)
    return X, z


def main():
    X, z = make_ensemble()

    # Continuous driver → the Fiedler eigenvector of the consensus recurrence
    # Laplacian. `store_adjacency_matrix` keeps A around so we can look at it.
    model = RecurrenceManifold(store_adjacency_matrix=True).fit(X)
    z_pred = model.labels_

    rho = abs(spearmanr(z_pred, z).correlation)
    print(f"recovered driver Spearman |rho| = {rho:.3f}")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4),
                                   gridspec_kw={"width_ratios": [2, 1]})
    plot_driver_overlay(z, z_pred, ax=ax0)
    ax0.set_title(f"driver reconstruction (|rho| = {rho:.2f})")
    plot_recurrence_matrix(model.adjacency_matrix, ax=ax1)
    ax1.set_title("consensus recurrence graph A")
    fig.tight_layout()

    out = os.path.join(HERE, "quickstart.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
