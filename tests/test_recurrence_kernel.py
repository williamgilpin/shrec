"""Regression and math-correctness tests for the exp-of-distance recurrence
kernel (`recurrence/kernel.py`).

- MM6 (§5b.1): the ensemble p-norm aggregation limits — `ord=1` is the
  elementwise mean of per-channel kernels; `ord → ∞` is the min-over-channels
  distance (the Sauer `inf_k d^(k)` the classical baseline approximates).
- `distance_to_connectivity` bracket fragility (same failure class as the
  `fit_rho_sigma` σ-solve stall): `root_scalar` assumed a sign change on a
  fixed bracket `[1e-16, dscale]` that does not exist for every distance
  distribution.
"""
import warnings

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from shrec.recurrence.kernel import data_to_connectivity, distance_to_connectivity


def _per_channel_kernels(X, scale=1.0):
    """Reconstruct the per-channel base affinities `a_i = exp(-surprise_i/thresh)`
    exactly as `data_to_connectivity` builds them, so the p-norm aggregation can
    be checked against a closed form. `data_to_connectivity(X, ord=p)` equals
    the power-mean `(mean_i a_i**p) ** (1/p)`."""
    thresh = np.median(
        np.linalg.norm(X - np.median(X, axis=1, keepdims=True), axis=-1)
    )
    thresh = 1.0 / (scale * thresh)
    kernels = []
    for Xi in X:
        dmat = cdist(Xi, Xi)
        surprise = dmat / np.std(dmat)
        kernels.append(np.exp(-surprise / thresh))
    return np.array(kernels)


class TestEnsemblePNormLimits:
    """MM6 (must) — the ensemble aggregation in `data_to_connectivity` is a
    power-mean of per-channel kernels: `bd = (mean_i a_i**ord) ** (1/ord)`.
    `ord=1` collapses to the arithmetic mean; `ord → ∞` to the elementwise max
    `= exp(-(min_i surprise_i)/thresh)`, i.e. the closest-channel (Sauer
    `inf_k`) recurrence.
    """

    @pytest.fixture
    def X(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal((4, 30, 2))  # 4 channels, T=30, D=2

    def test_ord_one_is_elementwise_mean(self, X):
        a = _per_channel_kernels(X)
        np.testing.assert_allclose(
            data_to_connectivity(X, ord=1.0), a.mean(axis=0), atol=1e-12,
        )

    def test_arbitrary_ord_is_power_mean(self, X):
        a = _per_channel_kernels(X)
        for p in (2.0, 3.5):
            np.testing.assert_allclose(
                data_to_connectivity(X, ord=p),
                (a ** p).mean(axis=0) ** (1.0 / p),
                atol=1e-10,
            )

    def test_large_ord_approaches_min_over_channels(self, X):
        a = _per_channel_kernels(X)
        elementwise_max = a.max(axis=0)  # = exp(-(min surprise)/thresh)
        # Restrict to well-conditioned entries: a_i**ord underflows to 0 once
        # a_i is small, so the L∞ limit is only meaningful where the closest
        # channel is appreciably recurrent. The power-mean also approaches the
        # max only as fast as 1/nb**(1/ord), so convergence is gradual.
        mask = elementwise_max > 0.5
        gaps = [
            np.abs(data_to_connectivity(X, ord=float(p))[mask] - elementwise_max[mask]).max()
            for p in (5, 20, 100)
        ]
        # Monotone decrease toward the elementwise max as ord grows.
        assert gaps[0] > gaps[1] > gaps[2]
        assert gaps[2] < 0.03


class TestDistanceToConnectivityBracket:
    """Pre-fix: small distance matrices raised
    `ValueError: f(a) and f(b) must have different signs` at the default
    sparsity, because the achievable mean-affinity floor (the diagonal alone
    forces mean ≥ 1/N) sits above the requested target, so no root is bracketed.
    """

    @pytest.mark.parametrize("n", [6, 8, 15, 40])
    def test_small_matrix_does_not_raise(self, n):
        rng = np.random.default_rng(n)
        # cdist(X, X) has a zero diagonal -> mean-affinity floor 1/N, which is
        # above the default target for these sizes. Pre-fix this raised
        # ValueError from the unbracketed root_scalar.
        X = rng.standard_normal((n, 2))
        dmat = cdist(X, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = distance_to_connectivity(dmat)
        assert out.shape == dmat.shape
        assert np.all(np.isfinite(out))

    def test_infeasible_sparsity_warns(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((8, 2))
        dmat = cdist(X, X)  # zero diagonal -> floor 1/8 > 1 - 0.99
        with pytest.warns(UserWarning, match="below the achievable floor"):
            distance_to_connectivity(dmat, sparsity=0.99)

    def test_feasible_sparsity_hits_target_mean_affinity(self):
        # Large enough that the 1/N floor is well below the target; the solver
        # must land on the requested mean mass (1 - sparsity).
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 2))
        dmat = cdist(X, X)
        sparsity = 0.9
        out = distance_to_connectivity(dmat, sparsity=sparsity)
        assert out.mean() == pytest.approx(1 - sparsity, abs=1e-3)

    def test_dscale_path_is_unchanged(self):
        dmat = cdist(np.arange(10)[:, None], np.arange(10)[:, None]).astype(float)
        np.testing.assert_allclose(
            distance_to_connectivity(dmat, dscale=2.0), np.exp(-dmat / 2.0),
        )


class TestConstantResponseRejection:
    """MM17 (§5b.2) — `data_to_connectivity` drops constant channels with a
    warning before building the kernel. A constant response carries no
    recurrence information (its distance matrix is all zeros → degenerate
    `surprise = 0/0`), so silently including it would inject NaNs / a uniform
    block into the consensus. The filter must (a) warn and (b) produce exactly
    the result of the surviving non-constant channels.
    """

    def _channels(self, seed=0):
        rng = np.random.default_rng(seed)
        good = rng.standard_normal((2, 25, 3))           # two informative channels
        const = np.tile(rng.standard_normal((1, 1, 3)), (1, 25, 1))  # one flat channel
        return good, const

    def test_constant_channel_is_warned_and_dropped(self):
        good, const = self._channels()
        stacked = np.concatenate([good, const], axis=0)

        with pytest.warns(UserWarning, match="Constant time series"):
            out_with_const = data_to_connectivity(stacked)
        out_good_only = data_to_connectivity(good)

        np.testing.assert_allclose(out_with_const, out_good_only, atol=1e-12)
        assert np.all(np.isfinite(out_with_const))

    def test_no_warning_when_all_channels_vary(self):
        good, _ = self._channels()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            data_to_connectivity(good)
