"""Connectivity stage strategies — embedded response stack → consensus affinity A.

A `Connectivity` is the first half of the SHREC pipeline (paper Appendix B
steps 2–4): it turns the embedded response ensemble `(N, T, D)` into a single
`(T, T)` recurrence graph `A`. Swapping the connectivity strategy is how you
choose *how* recurrences are built and aggregated, independent of *how* the
driver is then reconstructed from `A` (see `shrec.reconstruct`).

See `docs/decisions/0008-pipeline-modularization.md`.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist

from shrec.recurrence.consensus import data_to_connectivity2
from shrec.recurrence.kernel import data_to_connectivity
from shrec.utils import common_neighbors_ratio, sparsify


class Connectivity(ABC):
    """Map an embedded response stack `(N, T, D)` to a consensus affinity
    matrix `(T, T)`."""

    @abstractmethod
    def __call__(self, X):
        ...


class SimplicialConnectivity(Connectivity):
    """Canonical SHREC connectivity: per-response adaptive fuzzy simplicial
    complex (`ρ_i, σ_i` root-solve) + consensus aggregation across responses.

    The defaults reproduce the historical `data_to_connectivity2(X,
    time_exclude=...)` call used by the canonical models exactly
    (`k=10, tol=1e-5, aggregation="mean", metric="euclidean"`).

    With ``sparsify=True`` the dense consensus affinity is thresholded to the
    target sparsity `1 − tolerance`, keeping only the strongest recurrences.
    This is what makes the adaptive graph consumable by a reconstructor that
    needs a sparse/binary input (e.g. `UnionFindReconstructor`): a dense fuzzy
    affinity otherwise makes every node mutually reachable, collapsing the
    Sauer equivalence classes to a single component (decision 0009).
    """

    def __init__(self, k=10, tol=1e-5, time_exclude=0, aggregation="mean",
                 metric="euclidean", sparsify=False, tolerance=0.01,
                 weighted=True, verbose=False):
        self.k = k
        self.tol = tol
        self.time_exclude = time_exclude
        self.aggregation = aggregation
        self.metric = metric
        self.sparsify = sparsify
        self.tolerance = tolerance
        self.weighted = weighted
        self.verbose = verbose

    def __call__(self, X):
        A = data_to_connectivity2(
            X, k=self.k, tol=self.tol, time_exclude=self.time_exclude,
            aggregation=self.aggregation, metric=self.metric,
            verbose=self.verbose,
        )
        if self.sparsify:
            A = sparsify(A, 1 - self.tolerance, weighted=self.weighted)
        return A


class ExpKernelConnectivity(Connectivity):
    """Sauer-style fixed-scale exp-of-distance recurrence kernel, with an
    optional sparsify-to-adjacency step.

    `order` is the p-norm ensemble aggregation exponent (large → Sauer
    `inf_k`); this is the knob the dead base-class `aggregation_order`
    parameter was meant to expose. With ``sparsify=True`` the output is
    thresholded to the target sparsity `1 − tolerance` — the form the
    classical union-find baseline consumes.
    """

    def __init__(self, scale=1.0, order=500.0, time_exclude=0, metric="euclidean",
                 sparsify=False, tolerance=0.01, weighted=True):
        self.scale = scale
        self.order = order
        self.time_exclude = time_exclude
        self.metric = metric
        self.sparsify = sparsify
        self.tolerance = tolerance
        self.weighted = weighted

    def __call__(self, X):
        bd = data_to_connectivity(
            X, time_exclude=self.time_exclude, scale=self.scale,
            ord=self.order, metric=self.metric,
        )
        if self.sparsify:
            bd = sparsify(bd, 1 - self.tolerance, weighted=self.weighted)
        return bd


class CommonNeighborsConnectivity(Connectivity):
    """Hirata–Nomura recurrence similarity: per-response binary recurrences at
    an absolute distance percentile, summed and binarised across responses,
    then re-weighted by the common-neighbour ratio (Nomura et al. 2022)."""

    def __init__(self, percentile=0.1, metric="euclidean"):
        self.percentile = percentile
        self.metric = metric

    def __call__(self, X):
        amat = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            dmat = cdist(X[i], X[i], metric=self.metric)
            thresh = np.percentile(dmat, self.percentile)
            amat += (dmat <= thresh).astype(int)
        amat[amat > 0] = 1
        return common_neighbors_ratio(amat)


class PrecomputedConnectivity(Connectivity):
    """Return a fixed, precomputed affinity matrix, ignoring the input.

    The injection seam for power users and tests: drive the reconstruction
    stage from a known `A` without going through the recurrence machinery
    (replaces the old `monkeypatch.setattr(..., data_to_connectivity2)`)."""

    def __init__(self, A):
        self.A = np.asarray(A)

    def __call__(self, X):
        return self.A
