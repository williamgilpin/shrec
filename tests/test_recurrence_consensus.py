"""Math-correctness tests for consensus aggregation and metric plumbing
(docs/tests-math.md; plan 0009). There was no dedicated consensus test module
before the 0008 refactor exposed the `aggregation` / `metric` knobs.

- **MM38** — `SimplicialConnectivity(aggregation=…)` limits: `pnorm:1 ≡ mean`,
  `pnorm:p → max` as `p → ∞`, and `max` is the elementwise max of the
  per-response stack. The simplicial-consensus analogue of MM6 (which pins the
  exp-kernel p-norm).
- **MM39** — distance `metric` plumbing: `metric` reaches `cdist`, so a
  non-euclidean metric changes the affinity (the knob isn't silently ignored),
  while `metric="euclidean"` reproduces the default — and the MM3/MM4
  invariants (unit diagonal, symmetry) survive any metric.
"""
import numpy as np
import pytest

from shrec.recurrence import data_to_connectivity2
from shrec.recurrence.consensus import _aggregate


def _stack(seed=0, k=4, n=6):
    """A stack of `k` random affinity-like matrices in [0, 1]."""
    rng = np.random.default_rng(seed)
    return [rng.random((n, n)) for _ in range(k)]


# --- MM38 — aggregation limits ----------------------------------------------

class TestAggregationLimits:
    """MM38 — the consensus aggregator's closed-form limits."""

    def test_pnorm_one_equals_mean(self):
        mats = _stack()
        np.testing.assert_allclose(
            _aggregate(list(mats), len(mats), "pnorm:1"),
            _aggregate(list(mats), len(mats), "mean"),
            atol=1e-12,
        )

    def test_mean_is_the_arithmetic_mean(self):
        mats = _stack()
        np.testing.assert_allclose(
            _aggregate(list(mats), len(mats), "mean"),
            np.mean(np.stack(mats), axis=0),
            atol=1e-12,
        )

    def test_max_is_elementwise_max(self):
        mats = _stack()
        np.testing.assert_allclose(
            _aggregate(list(mats), len(mats), "max"),
            np.max(np.stack(mats), axis=0),
            atol=1e-12,
        )

    def test_pnorm_approaches_max_for_large_p(self):
        mats = _stack()
        true_max = np.max(np.stack(mats), axis=0)
        approx = _aggregate(list(mats), len(mats), "pnorm:80")
        # The power mean rises to the max from below as p → ∞. On well-
        # conditioned entries (max bounded away from 0) p=80 is already tight;
        # tiny-max entries converge slower, so restrict the comparison there
        # (same underflow caveat as MM6).
        big = true_max > 0.3
        assert np.all(approx <= true_max + 1e-9)
        np.testing.assert_allclose(approx[big], true_max[big], rtol=0.05)

    def test_unknown_aggregation_raises(self):
        mats = _stack()
        with pytest.raises(ValueError, match="Unknown aggregation"):
            _aggregate(list(mats), len(mats), "median")

    def test_pnorm_one_equals_mean_through_full_pipeline(self, rng):
        """The wiring, not just the helper: `data_to_connectivity2` with
        `pnorm:1` reproduces the default mean consensus on a real ensemble."""
        X = rng.standard_normal((4, 30, 2))
        np.testing.assert_allclose(
            data_to_connectivity2(X, aggregation="pnorm:1"),
            data_to_connectivity2(X, aggregation="mean"),
            atol=1e-9,
        )


# --- MM39 — distance metric plumbing ----------------------------------------

class TestMetricPlumbing:
    """MM39 — `metric` reaches the per-response `cdist`."""

    def test_euclidean_is_the_default(self, rng):
        X = rng.standard_normal((4, 30, 2))
        np.testing.assert_allclose(
            data_to_connectivity2(X, metric="euclidean"),
            data_to_connectivity2(X),
            atol=1e-12,
        )

    def test_non_euclidean_metric_changes_affinity(self, rng):
        X = rng.standard_normal((4, 30, 2))
        A_eucl = data_to_connectivity2(X, metric="euclidean")
        A_city = data_to_connectivity2(X, metric="cityblock")
        # The knob is not silently ignored.
        assert not np.allclose(A_eucl, A_city)

    def test_invariants_hold_under_any_metric(self, rng):
        X = rng.standard_normal((4, 30, 2))
        for metric in ("euclidean", "cityblock", "cosine"):
            A = data_to_connectivity2(X, metric=metric)
            np.testing.assert_allclose(A, A.T, atol=1e-12)          # MM4 symmetry
            np.testing.assert_allclose(np.diag(A), 1.0, atol=1e-12)  # MM3 unit diagonal
            assert A.min() >= -1e-12 and A.max() <= 1.0 + 1e-12
