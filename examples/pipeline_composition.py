"""Composing the SHREC pipeline from stages (plan 0009 / E2).

The headline example for the 0008 stage-strategy refactor: the four named
models are *presets* over `ShrecPipeline(connectivity=..., reconstructor=...)`,
and you can swap either stage independently. Run:

    uv run python examples/pipeline_composition.py

Writes `examples/pipeline_composition.png`.

Sensible-combo note: `Fiedler`/`Leiden` consume an affinity (any
connectivity); `UnionFind` needs a sparse/binary graph; `Isomap` wants a
dissimilarity. This example sticks to the spectral reconstructor and swaps the
*connectivity* and the *aggregation*, which are the meaningful continuous-driver
knobs.
"""
import os

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shrec import RecurrenceManifold, ShrecPipeline
from shrec.recurrence.connectivity import (
    ExpKernelConnectivity,
    PrecomputedConnectivity,
    SimplicialConnectivity,
)
from shrec.reconstruct import FiedlerReconstructor
from shrec.plotting import plot_driver_overlay

from quickstart import make_ensemble  # reuse the synthetic ensemble

HERE = os.path.dirname(__file__)


def rho(z_pred, z):
    return abs(spearmanr(z_pred, z).correlation)


def main():
    X, z = make_ensemble()

    # 1. The preset and the explicit composition are the same thing.
    preset = RecurrenceManifold().fit(X).labels_
    explicit = ShrecPipeline(
        connectivity=SimplicialConnectivity(),
        reconstructor=FiedlerReconstructor(),
    ).fit(X).labels_
    agree = abs(np.dot(
        preset / np.linalg.norm(preset), explicit / np.linalg.norm(explicit)))
    print(f"RecurrenceManifold == ShrecPipeline(Simplicial, Fiedler): |cos|={agree:.4f}")

    # 2. Swap the connectivity under the same (Fiedler) reconstructor.
    runs = [
        ("Simplicial + Fiedler\n(= RecurrenceManifold)",
         SimplicialConnectivity(), FiedlerReconstructor()),
        ("ExpKernel + Fiedler\n(Sauer kernel, swapped in)",
         ExpKernelConnectivity(sparsify=False), FiedlerReconstructor()),
        ("Simplicial(max-consensus) + Fiedler\n(aggregation knob)",
         SimplicialConnectivity(aggregation="max"), FiedlerReconstructor()),
    ]

    fig, axes = plt.subplots(len(runs) + 1, 1, figsize=(9, 3 * (len(runs) + 1)))
    for ax, (title, conn, rec) in zip(axes, runs):
        v = ShrecPipeline(connectivity=conn, reconstructor=rec).fit(X).labels_
        plot_driver_overlay(z, v, ax=ax)
        ax.set_title(f"{title}   |rho|={rho(v, z):.2f}")

    # 3. Drive reconstruction from a user-supplied graph via PrecomputedConnectivity.
    A = RecurrenceManifold(store_adjacency_matrix=True).fit(X).adjacency_matrix
    v = ShrecPipeline(
        connectivity=PrecomputedConnectivity(A),
        reconstructor=FiedlerReconstructor(),
        standardize=False,
    ).fit(np.zeros((A.shape[0], 1))).labels_
    plot_driver_overlay(z, v, ax=axes[-1])
    axes[-1].set_title(f"PrecomputedConnectivity(A) + Fiedler   |rho|={rho(v, z):.2f}")

    fig.tight_layout()
    out = os.path.join(HERE, "pipeline_composition.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
