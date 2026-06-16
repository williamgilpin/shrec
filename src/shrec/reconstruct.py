"""Reconstructor stage strategies — consensus affinity A → driver labels.

A `Reconstructor` is the second half of the SHREC pipeline (paper Appendix B
step 5): it turns the recurrence graph `A` `(T, T)` into the reconstructed
driver — a discrete label per timepoint (community detection / union-find) or
a continuous coordinate (Fiedler eigenvector / Isomap embedding). Swapping the
reconstructor is independent of how `A` was built (see
`shrec.recurrence.connectivity`).

See `docs/decisions/0008-pipeline-modularization.md`.
"""
import warnings
from abc import ABC, abstractmethod

import numpy as np
import scipy.linalg
from sklearn.manifold import Isomap

from shrec.graph import _leiden, solve_union_find
from shrec.utils import allclose_len, nan_fill


class Reconstructor(ABC):
    """Map a consensus affinity matrix `(T, T)` to driver labels."""

    @abstractmethod
    def __call__(self, A):
        """Return `labels_` of shape `(T,)` or `(T, n_components)`,
        ordered by timepoint index."""

    def extra_attrs(self, labels):
        """Extra attributes the pipeline should set on the model after
        reconstruction (e.g. cluster counts). Default: none."""
        return {}


class LeidenReconstructor(Reconstructor):
    """Discrete driver — Leiden community detection on `A` (paper Appendix B
    step 5, discrete-time path). Lifts the body of the former
    `RecurrenceClustering.fit`."""

    def __init__(self, resolution=1.0, objective="modularity",
                 method="graspologic", random_state=None):
        self.resolution = resolution
        self.objective = objective
        self.method = method
        self.random_state = random_state

    def __call__(self, A):
        indices, labels = _leiden(
            A, resolution=self.resolution, objective=self.objective,
            method=self.method, random_state=self.random_state,
        )
        sort_inds = np.argsort(indices)
        indices, labels = indices[sort_inds], labels[sort_inds]

        out = -np.ones(A.shape[0], dtype=indices.dtype)
        out[indices] = labels
        return out

    def extra_attrs(self, labels):
        has_unclassified = bool(np.any(labels < 0))
        n_clusters = len(np.unique(labels)) - has_unclassified
        return {"has_unclassified": has_unclassified, "n_clusters": n_clusters}


class FiedlerReconstructor(Reconstructor):
    """Continuous driver — Fiedler eigenvector(s) of the graph Laplacian
    `L = D − A` (paper Appendix B step 5, continuous-time path). Lifts the
    spectral body of the former `RecurrenceManifold.fit`, including the
    disconnected-graph guard and the optional NCut normalisation."""

    def __init__(self, n_components=1, normalize_laplacian=False, verbose=False):
        self.n_components = n_components
        self.normalize_laplacian = normalize_laplacian
        self.verbose = verbose

    def __call__(self, A):
        affinity = np.asarray(A)
        degree = affinity.sum(axis=1)
        laplacian = np.diag(degree) - affinity

        if self.normalize_laplacian:
            # NCut / random-walk normalisation via the generalised eigenproblem
            # L v = λ D v (Shi–Malik): balances by volume rather than node
            # count, down-weighting high-degree responses. Generalised
            # eigenvalues live in [0, 2], so the connectivity scale is O(1).
            eigvals, eigvecs = scipy.linalg.eigh(
                laplacian, np.diag(degree), subset_by_index=[1, self.n_components],
            )
            connectivity_scale = 1.0
        else:
            eigvals, eigvecs = scipy.linalg.eigh(
                laplacian, subset_by_index=[1, self.n_components],
            )
            connectivity_scale = degree.sum()

        # λ₂ > 0 iff the graph is connected; a 0 has multiplicity = number of
        # connected components. When λ₂ ≈ 0 the returned eigenvector lies in a
        # degenerate near-null space and is a component indicator, not a smooth
        # driver coordinate. Threshold is scaled to the spectrum.
        if eigvals[0] <= 1e-10 * connectivity_scale:
            warnings.warn(
                "Consensus recurrence graph is (nearly) disconnected "
                f"(algebraic connectivity λ₂={eigvals[0]:.3e}); the Fiedler "
                "eigenvector may be a connected-component indicator rather "
                "than a smooth driver. Consider lowering `tolerance`/"
                "`time_exclude` or increasing the response ensemble size."
            )

        return nan_fill(eigvecs.squeeze())


class UnionFindReconstructor(Reconstructor):
    """Sauer (PRL 2004) equivalence-class union-find on a binarised recurrence
    graph — the noise-free discrete limit. Lifts the body of the former
    `ClassicalRecurrenceClustering.fit` (post-connectivity)."""

    def __call__(self, A):
        all_merged_inds = [np.sort(np.where(row)[0]) for row in A]
        merged_inds = solve_union_find([list(item) for item in all_merged_inds])
        merged_inds = [np.sort(np.array(item)) for item in merged_inds]

        known_items = []
        item_labels = []
        for item in merged_inds:
            add_flag = False
            for j, known_item in enumerate(known_items):
                if allclose_len(item, known_item):
                    item_labels.append(j)
                    add_flag = True
            if not add_flag:
                known_items.append(item)
                item_labels.append(1)
        return np.array(item_labels)


class IsomapReconstructor(Reconstructor):
    """Hirata–Nomura continuous driver — Isomap embedding of the precomputed
    common-neighbour similarity. Lifts the body of the former
    `HirataNomuraIsomap.fit` (post-connectivity)."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def __call__(self, A):
        iso = Isomap(n_components=self.n_components, metric='precomputed')
        return nan_fill(iso.fit_transform(A))
