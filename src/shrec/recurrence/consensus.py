"""Consensus aggregation of per-response affinity matrices (paper Appendix B step 4)."""
import warnings

import numpy as np

from shrec.recurrence.simplicial import dataset_to_simplex


def _aggregate(per_response, nb, aggregation):
    """Aggregate a stack of per-response affinities into one consensus matrix.

    ``per_response`` is an iterator/list of ``(T, T)`` affinity matrices.
    Aggregation modes:

    - ``"mean"`` — the paper's consensus (`A_ij = (1/K) Σ_k A^(k)_ij`).
      Streamed so the full ``(K, T, T)`` stack is never materialised.
    - ``"pnorm:<p>"`` — the generalised power mean `(mean_k a**p)**(1/p)`;
      ``p→∞`` approaches the elementwise max (Sauer `inf_k`), ``p=1`` the
      mean. Materialises the stack.
    - ``"max"`` — elementwise max over responses. Materialises the stack.
    """
    if aggregation == "mean":
        wmat = None
        for a in per_response:
            wmat = a / nb if wmat is None else wmat + a / nb
        return wmat

    stack = np.stack(list(per_response), axis=0)
    if aggregation == "max":
        return stack.max(axis=0)
    if aggregation.startswith("pnorm:"):
        p = float(aggregation.split(":", 1)[1])
        return np.mean(stack ** p, axis=0) ** (1.0 / p)
    raise ValueError(
        f"Unknown aggregation {aggregation!r}; expected 'mean', 'max', "
        "or 'pnorm:<p>'."
    )


def data_to_connectivity2(
    X, k=10, tol=1e-5, time_exclude=0, aggregation="mean", metric="euclidean",
    verbose=False,
):
    """
    Consensus fuzzy simplicial complex across an ensemble of responses.

    For each response in the ensemble, build the per-response adaptive
    recurrence affinity A^(k) via `dataset_to_simplex` (paper Appendix B
    step 3), then aggregate across responses (step 4). The default
    ``aggregation="mean"`` is the paper's consensus; ``"max"`` /
    ``"pnorm:<p>"`` are generalisations (the Sauer limit lives at large p).
    This is the canonical SHREC connectivity for the discrete/continuous
    driver models — the alternative `data_to_connectivity` is a simpler
    fixed-scale exp-of-distance kernel kept for the Sauer baseline.

    Args:
        X (np.ndarray): dataset of shape (n_responses, n_times, n_dims).
        k (int): number of nearest neighbours used per row in the
            simplicial-complex root-solve.
        tol (float): tolerance for the σ root-solver.
        time_exclude (int): if >0, zero entries within ±time_exclude of
            the diagonal to suppress trivial trajectory recurrences.
        aggregation (str): "mean" (default) | "max" | "pnorm:<p>".
        metric (str): distance metric passed to `cdist` per response.
        verbose (bool): whether to print progress.

    Returns:
        wmat (np.ndarray): consensus adjacency matrix of shape
            (n_times, n_times).
    """
    nb, nt, _ = X.shape

    if not verbose:
        warnings.filterwarnings('ignore')

    def per_response():
        for ind, X0 in enumerate(X):
            if verbose and nb >= 10 and ind % (nb // 10) == 0:
                print(ind, "/", len(X), flush=True)
            yield dataset_to_simplex(X0, k=k, tol=tol, metric=metric)

    wmat = _aggregate(per_response(), nb, aggregation)

    if time_exclude > 0:
        mask = 1 - (
            np.triu(np.ones_like(wmat), k=-time_exclude)
            * np.tril(np.ones_like(wmat), k=time_exclude)
        )
        wmat *= mask

    return wmat
