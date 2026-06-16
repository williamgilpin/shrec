"""Per-response fuzzy simplicial complex (paper Appendix B step 3).

For each point, find its k nearest neighbours, set `ρ_i` to the closest-
neighbour distance, then root-solve for `σ_i` such that
`Σ_m exp(-ReLU(d_im − ρ_i)/σ_i) = log₂(k)`. The resulting per-row
affinity is symmetrised via the fuzzy union `A + Aᵀ − A∘Aᵀ`.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import cdist


def relu(x):
    return np.maximum(0, x)


def fit_rho_sigma(knn_dists, k, tol=1e-12):
    """Solve Appendix-B step 3 for one point's local connectivity parameters.

    Args:
        knn_dists (np.ndarray): the point's k nearest-neighbour distances,
            sorted ascending (so ``knn_dists[0]`` is the closest neighbour).
        k (int): neighbour count setting the target mass ``log2(k)``.
        tol (float): brentq absolute tolerance on σ (the root location).

    Returns:
        (rho, sigma): ``rho`` is the nearest-neighbour distance; ``sigma``
        root-solves ``Σ_m exp(-ReLU(d_m - rho)/sigma) = log2(k)``.
    """
    rho = knn_dists[0]

    def func(sig):
        return sum(np.exp(-relu(knn_dists - rho) / sig)) - np.log2(k)

    # func is strictly increasing in σ: each term exp(-x/σ) rises from 0 (x>0)
    # or 1 (x=0) at σ→0⁺ to 1 at σ→∞. Generically it crosses zero once, so
    # brentq on a wide bracket finds the root robustly where fsolve's step-size
    # convergence test silently stalls at the initial guess on a fraction of
    # rows. The exception is a fully-tied neighbourhood (all k nearest at the
    # same distance, e.g. from clipped/quantised series): the mass is pinned at
    # k > log₂k for every σ, so no root exists. σ is then indeterminate for the
    # k-NN; fall back to ρ (scale-covariant, sets the decay scale for the tail).
    lo, hi = 1e-12, 1e6
    if func(lo) >= 0:
        return rho, rho
    sigma = brentq(func, lo, hi, xtol=tol)
    return rho, sigma


def dataset_to_simplex(X, k=20, tol=1e-12, precomputed=False, metric="euclidean"):
    """
    Fuzzy simplicial complex over a single point cloud.

    Args:
        X (np.ndarray): dataset of shape (n_samples, n_features) or, if
            ``precomputed=True``, a precomputed (n, n) distance matrix.
        k (int): number of nearest neighbours used in the σ root-solve.
        tol (float): tolerance passed to `scipy.optimize.fsolve`.
        precomputed (bool): treat ``X`` as a distance matrix.
        metric (str): any metric accepted by
            `scipy.spatial.distance.cdist` (ignored when ``precomputed``).

    Returns:
        wmat (np.ndarray): affinity matrix of shape (n_samples, n_samples).

    References:
        McInnes, Healy, Melville. "UMAP: Uniform Manifold Approximation
        and Projection for Dimension Reduction." arXiv:1802.03426 (2018).
    """
    dmat = X.copy() if precomputed else cdist(X, X, metric=metric)

    n = dmat.shape[0]
    dmat_zerofilled = dmat.copy()
    # Send the self-distance (and any exact duplicates) to +inf so they sort to
    # the end and are excluded from the neighbourhood. Index 0 of the sorted row
    # is then the *nearest* distinct neighbour — so take [:k] (the k nearest),
    # not [1:k+1]. Per Appendix B, ρ_i = closest-neighbour distance = dists[0];
    # the old [1:k+1] slice double-skipped (inf already removed self) and set ρ
    # to the *second*-nearest neighbour. See MM5 in docs/tests-math.md.
    dmat_zerofilled[dmat_zerofilled < 1e-10] = np.inf
    dists = np.partition(dmat_zerofilled, k, axis=1)
    dists = np.sort(dists, axis=1)[:, :k]
    for i in range(n):
        rho_i, sigma_i = fit_rho_sigma(dists[i], k, tol=tol)
        dmat[i] = np.exp(-relu(dmat[i] - rho_i) / sigma_i)
    return dmat + dmat.T - dmat * dmat.T
