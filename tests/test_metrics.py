"""Oracle for the quantile sparsifier (docs/tests-math.md §5b.1 MM9).

``utils/metrics.py:sparsify`` binarises a matrix by thresholding |a| at the
``sparsity``-quantile (``np.percentile(..., interpolation="higher")``), zeroing
everything at or below the threshold. Two contract properties:

- **achieved ≥ requested** — the docstring promises the realised zero-fraction
  is *at least* the requested ``sparsity`` (the "higher" interpolation rounds
  the threshold up so ties never undershoot the target).
- **idempotence** — applying the sparsifier twice at the same target equals
  applying it once. The first pass produces a binary matrix whose zero-fraction
  ≥ ``sparsity``; the second pass's quantile then lands inside the zero block,
  so the threshold is 0 and the binary pattern is a fixed point.
"""
import numpy as np
import pytest

from shrec.utils.metrics import sparsify


class TestSparsifyQuantile:

    @pytest.mark.parametrize("sparsity", [0.5, 0.61, 0.77, 0.9])
    @pytest.mark.parametrize("seed", range(4))
    def test_achieved_sparsity_at_least_requested(self, sparsity, seed):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((40, 40))  # continuous → no ties
        out = sparsify(a, sparsity=sparsity)
        achieved = np.mean(out == 0)
        assert achieved >= sparsity - 1e-12

    @pytest.mark.parametrize("sparsity", [0.5, 0.61, 0.77, 0.9])
    @pytest.mark.parametrize("seed", range(4))
    def test_idempotent_at_same_threshold(self, sparsity, seed):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((40, 40))
        once = sparsify(a, sparsity=sparsity)
        twice = sparsify(once, sparsity=sparsity)
        np.testing.assert_array_equal(twice, once)

    def test_output_is_binary_when_unweighted(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((20, 20))
        out = sparsify(a, sparsity=0.7)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_more_sparsity_zeros_more_entries(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((50, 50))
        zeros = [np.mean(sparsify(a, sparsity=s) == 0) for s in (0.3, 0.6, 0.9)]
        assert zeros[0] <= zeros[1] <= zeros[2]
