"""Sanity oracle for the Hirata-Nomura Isomap baseline (docs/tests-math.md
§5b.4 MM29).

`HirataNomuraIsomap` is the comparison baseline: a common-neighbour-ratio
consensus similarity (Nomura 2022) embedded with Isomap (Hirata 2008). Isomap
with `metric='precomputed'` requires its input to be a valid dissimilarity
matrix — symmetric, zero on the diagonal, non-negative. MM29 pins that the
consensus matrix the model feeds Isomap satisfies that contract, and that `fit`
produces a finite embedding of the right shape. (The element-wise correctness of
`common_neighbors_ratio` itself is MM12 in `tests/test_graph_adjacency.py`; this
test guards the property *at the baseline-model boundary*.)
"""
import numpy as np
import pytest

from shrec.models import HirataNomuraIsomap


def _square_wave_responses(T=300, n_responses=6, seed=0):
    rng = np.random.default_rng(seed)
    z = np.where(np.arange(T) % 2 == 0, 0.2, 0.8)
    r_values = rng.uniform(3.81, 3.97, size=n_responses)
    X = np.empty((T, n_responses))
    for k, r in enumerate(r_values):
        x = np.empty(T)
        x[0] = rng.uniform(0.1, 0.9)
        for t in range(T - 1):
            x[t + 1] = np.clip(r * x[t] * (1 - x[t]) + 0.5 * z[t], 0.0, 1.0)
        X[:, k] = x
    return X


class TestHirataNomuraConsensusMatrix:

    def test_consensus_matrix_is_a_valid_dissimilarity(self):
        X = _square_wave_responses()
        model = HirataNomuraIsomap(n_components=2, store_adjacency_matrix=True)
        model.fit(X)
        W = model.adjacency_matrix

        np.testing.assert_allclose(W, W.T, atol=1e-12)        # symmetric
        np.testing.assert_allclose(np.diag(W), 0.0, atol=1e-12)  # zero diagonal
        assert np.all(W >= -1e-12)                            # non-negative
        assert np.all(W <= 1.0 + 1e-12)                       # ratio in [0, 1]

    def test_fit_produces_finite_embedding_of_expected_shape(self):
        X = _square_wave_responses(T=300)
        model = HirataNomuraIsomap(n_components=2).fit(X)
        assert model.labels_.shape == (300, 2)
        assert np.all(np.isfinite(model.labels_))
