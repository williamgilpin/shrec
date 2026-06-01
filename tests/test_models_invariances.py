"""Pipeline-level invariance tests (docs/tests-math.md §5b.2).

Properties the whole `fit_predict` pipeline must satisfy by symmetry
arguments — independent of the algorithm details. Drift in any of these
indicates a real bug.
"""
import numpy as np
import pytest

graspologic = pytest.importorskip("graspologic")

from sklearn.metrics import adjusted_rand_score

from shrec.models import RecurrenceClustering, RecurrenceManifold


def _square_wave_responses(T=400, n_responses=8, seed=0):
    """Period-2 driver with a small logistic ensemble — fast enough for
    the per-property tests to run twice within a single test invocation.
    """
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


# --- §5b.2 MM13 — response-permutation invariance --------------------------

class TestResponsePermutationInvariance:
    """The order of response channels carries no information. Permuting
    them must not change the discrete labels (up to label-permutation
    equivalence) or the continuous embedding (up to sign).
    """

    def test_clustering_is_permutation_invariant(self):
        X = _square_wave_responses(T=400, n_responses=8)
        rng = np.random.default_rng(123)
        perm = rng.permutation(X.shape[1])

        labels_a = RecurrenceClustering(random_state=1).fit(X).labels_
        labels_b = RecurrenceClustering(random_state=1).fit(X[:, perm]).labels_

        assert adjusted_rand_score(labels_a, labels_b) == pytest.approx(1.0, abs=1e-9)

    def test_manifold_is_permutation_invariant(self):
        X = _square_wave_responses(T=300, n_responses=8)
        rng = np.random.default_rng(123)
        perm = rng.permutation(X.shape[1])

        labels_a = RecurrenceManifold(random_state=1).fit(X).labels_
        labels_b = RecurrenceManifold(random_state=1).fit(X[:, perm]).labels_

        # Sign of Fiedler eigenvector is arbitrary; compare |cos|
        labels_a = labels_a / (np.linalg.norm(labels_a) + 1e-12)
        labels_b = labels_b / (np.linalg.norm(labels_b) + 1e-12)
        assert abs(float(np.dot(labels_a, labels_b))) > 0.999


# --- §5b.2 MM14 — time-reversal *approximate* symmetry ---------------------

class TestTimeReversalApproximateSymmetry:
    """The doc's MM14 claim (`A(X[::-1]) == A(X)[::-1, ::-1]` *exactly*) is
    too strong once delay embedding enters the picture: forward embedded
    points at time t carry past lags `(x_t, x_{t-1}, x_{t-2})`, while the
    reversed series at the corresponding time carries *future* lags. For
    a stationary process these are statistically equivalent — so the
    cluster structure survives — but the pairwise distance matrices are
    not byte-equal. These tests pin the approximate version, which is
    still a strong regression guard against directional leaks.
    """

    def test_clustering_under_time_reversal_is_approximate(self):
        X = _square_wave_responses(T=400, n_responses=8)

        forward = RecurrenceClustering(random_state=1).fit(X).labels_
        reverse = RecurrenceClustering(random_state=1).fit(X[::-1]).labels_

        assert adjusted_rand_score(forward[::-1], reverse) > 0.85

    def test_manifold_under_time_reversal_is_approximate(self):
        X = _square_wave_responses(T=300, n_responses=8)

        forward = RecurrenceManifold(random_state=1).fit(X).labels_
        reverse = RecurrenceManifold(random_state=1).fit(X[::-1]).labels_

        forward = forward[::-1] / (np.linalg.norm(forward) + 1e-12)
        reverse = reverse / (np.linalg.norm(reverse) + 1e-12)
        assert abs(float(np.dot(forward, reverse))) > 0.7


# --- §5b.2 MM15 — full-pipeline scale (per-channel affine) invariance -------

class TestFullPipelineScaleInvariance:
    """`standardize=True` (the default) z-scores each response, so the whole
    pipeline is invariant to an arbitrary *per-channel* affine transform
    `x_k → a_k x_k + b_k` (a_k > 0). This lifts the affinity-level scale
    oracle MM2 to the full `fit_predict`: MM2 (affinity unchanged under a
    global scale) + standardisation (removes per-channel scale and offset)
    = full invariance. A regression here means standardisation was bypassed
    or applied across the wrong axis.
    """

    def test_clustering_invariant_to_per_channel_affine(self):
        X = _square_wave_responses(T=400, n_responses=8)
        rng = np.random.default_rng(7)
        a = rng.uniform(0.1, 10.0, size=X.shape[1])
        b = rng.uniform(-5.0, 5.0, size=X.shape[1])
        X_affine = X * a + b

        labels_a = RecurrenceClustering(random_state=1).fit(X).labels_
        labels_b = RecurrenceClustering(random_state=1).fit(X_affine).labels_
        assert adjusted_rand_score(labels_a, labels_b) == pytest.approx(1.0, abs=1e-9)

    def test_manifold_invariant_to_per_channel_affine(self):
        X = _square_wave_responses(T=300, n_responses=8)
        rng = np.random.default_rng(7)
        a = rng.uniform(0.1, 10.0, size=X.shape[1])
        b = rng.uniform(-5.0, 5.0, size=X.shape[1])
        X_affine = X * a + b

        va = RecurrenceManifold(random_state=1).fit(X).labels_
        vb = RecurrenceManifold(random_state=1).fit(X_affine).labels_
        va = va / (np.linalg.norm(va) + 1e-12)
        vb = vb / (np.linalg.norm(vb) + 1e-12)
        assert abs(float(np.dot(va, vb))) > 0.999


# --- §5b.2 MM16 — idempotence under response duplication --------------------

class TestResponseDuplicationIdempotence:
    """Consensus aggregation is a *mean* over responses, so duplicating the
    whole ensemble cannot change the consensus graph: with k copies the mean is
    `(k·ΣA)/(k·N) = (ΣA)/N`, identical to one copy. Therefore `fit(X)` and
    `fit([X, X])` give the same driver. Pins that the consensus step weights
    responses uniformly and renormalises by the response count (rather than
    accumulating raw mass). Stated on the *ensemble* rather than a single
    repeated channel because `standardize_ts` squeezes an (T, 1) input to 1-D.
    """

    def test_clustering_idempotent_under_duplication(self):
        X = _square_wave_responses(T=400, n_responses=4)
        X_dup = np.concatenate([X, X], axis=1)  # two copies of the ensemble

        labels_one = RecurrenceClustering(random_state=1).fit(X).labels_
        labels_dup = RecurrenceClustering(random_state=1).fit(X_dup).labels_
        assert adjusted_rand_score(labels_one, labels_dup) == pytest.approx(1.0, abs=1e-9)

    def test_manifold_idempotent_under_duplication(self):
        X = _square_wave_responses(T=300, n_responses=4)
        X_dup = np.concatenate([X, X], axis=1)

        v_one = RecurrenceManifold(random_state=1).fit(X).labels_
        v_dup = RecurrenceManifold(random_state=1).fit(X_dup).labels_
        v_one = v_one / (np.linalg.norm(v_one) + 1e-12)
        v_dup = v_dup / (np.linalg.norm(v_dup) + 1e-12)
        assert abs(float(np.dot(v_one, v_dup))) > 0.999
