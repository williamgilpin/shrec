"""Geometry oracles for the recurrence distance primitive (docs/tests-math.md
§5b.1 MM7, MM8).

The whole SHREC recurrence step is built on Euclidean ``cdist`` between
delay-embedded states. Two facts about that primitive are load-bearing and
worth pinning analytically:

- **MM7** — recurrence is a property of the *shape* of the trajectory, not of
  the orientation or origin of the embedding space. So the per-response
  affinity ``dataset_to_simplex`` must be invariant under any rigid motion
  (rotation/reflection + translation) of the point cloud. This is tested at
  the affinity level (not just on ``cdist``) so it exercises the real ρ/σ
  pipeline, complementing the scale-invariance oracle MM2.
- **MM8** — the chosen metric is a true metric: ``d(i,k) ≤ d(i,j) + d(j,k)``.
  The ρ-as-nearest-neighbour / σ-bandwidth logic in ``fit_rho_sigma`` assumes
  nonnegative distances with a zero, minimal self-distance; a non-metric here
  would silently break those assumptions.
"""
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from shrec.recurrence.simplicial import dataset_to_simplex


def _random_orthogonal(d, rng):
    """A Haar-ish random orthogonal matrix (includes reflections) via QR."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    # Fix the sign ambiguity of QR so q is genuinely orthogonal with unit
    # columns; reflections are intentionally allowed (still an isometry).
    return q * np.sign(np.diag(r))


# --- §5b.1 MM7 — rigid-motion invariance of the affinity --------------------

class TestRecurrenceIsometryInvariance:

    @pytest.mark.parametrize("seed", range(5))
    def test_affinity_invariant_under_rotation_and_translation(self, seed):
        rng = np.random.default_rng(seed)
        n, d, k = 40, 4, 10
        X = rng.standard_normal((n, d))
        Q = _random_orthogonal(d, rng)
        b = rng.standard_normal(d)
        X_moved = X @ Q + b  # rigid motion: rotation/reflection + translation

        A = dataset_to_simplex(X, k=k)
        A_moved = dataset_to_simplex(X_moved, k=k)
        np.testing.assert_allclose(A_moved, A, atol=1e-6, rtol=1e-5)

    def test_pure_cdist_is_isometry_exact(self):
        # The geometric fact MM7 rests on, isolated: pairwise distances are
        # exactly preserved by a rigid motion (to floating point).
        rng = np.random.default_rng(7)
        X = rng.standard_normal((30, 5))
        Q = _random_orthogonal(5, rng)
        b = rng.standard_normal(5)
        np.testing.assert_allclose(cdist(X @ Q + b, X @ Q + b), cdist(X, X),
                                   atol=1e-10)


# --- §5b.1 MM8 — triangle inequality of the metric -------------------------

class TestMetricTriangleInequality:

    @pytest.mark.parametrize("seed", range(5))
    def test_triangle_inequality_on_all_triples(self, seed):
        rng = np.random.default_rng(seed)
        P = rng.standard_normal((25, 4))
        D = cdist(P, P)
        # d(i,k) ≤ d(i,j) + d(j,k) for every (i, j, k).
        lhs = D[:, None, :]            # d(i, k)
        rhs = D[:, :, None] + D.T[None, :, :]  # d(i, j) + d(j, k)
        assert np.all(lhs <= rhs + 1e-9)

    def test_zero_and_symmetry(self):
        rng = np.random.default_rng(8)
        P = rng.standard_normal((12, 3))
        D = cdist(P, P)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        assert np.all(D >= 0.0)
