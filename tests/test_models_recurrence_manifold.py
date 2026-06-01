"""Math-correctness tests for `RecurrenceManifold` (continuous driver).

Tied to docs/tests-math.md §5b.3 — **MM22** (block-diagonal Fiedler oracle)
is one of the four "must" tests driving the canonical-algorithm PR
(§6 step 2). It is expected to FAIL against the current implementation,
which uses `TruncatedSVD(A)` instead of the Fiedler eigenvector of
`L = D − A`. Locking in the fix flips this test green.
"""
import warnings

import numpy as np
import pytest

from shrec.models.models import RecurrenceManifold


def _two_block_affinity(p1=10, p2=30, bridge=1e-3):
    """K=2 weakly-connected blocks of **unequal size**, with unnormalised
    all-ones blocks (rows in block_i sum to ≈ p_i, so the graph is
    *irregular*). A single bridge edge of weight `bridge` connects the
    two blocks.

    The irregularity matters: for a regular graph A and L = D − A share
    eigenvectors, so SVD and Fiedler agree. By using row sums that differ
    by block, the top-2 right singular vectors of A become the per-block
    indicators (pure block 1 / pure block 2), while the Fiedler vector of
    L is the *block-difference* (zero-mean) vector spanning both blocks.
    The two answers measurably diverge — `|corr| ≈ 0.866` vs `1.0` — so
    the test fails sharply under the current `TruncatedSVD` path and
    passes after the §6 step-2 fix.
    """
    N = p1 + p2
    A = np.zeros((N, N))
    A[:p1, :p1] = np.ones((p1, p1))
    A[p1:, p1:] = np.ones((p2, p2))
    A[p1 - 1, p1] = bridge
    A[p1, p1 - 1] = bridge
    return A


def _analytical_fiedler(p1=10, p2=30):
    """For two unequal blocks, the Fiedler vector is constant on each
    block with opposite signs, weighted so the vector is zero-mean and
    unit-norm. The small bridge perturbation does not change this to
    leading order.
    """
    N = p1 + p2
    v = np.empty(N)
    # a, b chosen so p1*a + p2*b = 0 (zero mean) and ‖v‖₂ = 1
    b = -1.0
    a = -p2 * b / p1
    v[:p1] = a
    v[p1:] = b
    return v / np.linalg.norm(v)


class TestFiedlerEigenvectorOracle:
    """§5b.3 MM22 — `RecurrenceManifold` must extract the Fiedler eigenvector
    of `L = D − A`, not the second right singular vector of `A`. On a
    block-structured affinity matrix the two answers diverge to ~|corr| ≈
    0.7 vs 1.0, so this test fails sharply under the current code and is
    the canonical regression guard once the fix lands.
    """

    def test_two_block_affinity_recovers_block_indicator(self, monkeypatch):
        p1, p2 = 10, 30
        A = _two_block_affinity(p1=p1, p2=p2, bridge=1e-3)
        fiedler = _analytical_fiedler(p1=p1, p2=p2)

        # Bypass the time-series preprocessing — feed the precomputed
        # affinity straight into the spectral step. After the §6 step-4
        # split, `RecurrenceManifold` imports `data_to_connectivity2`
        # from `shrec.recurrence` directly, so patch the binding in the
        # model's own module rather than the legacy `shrec.models.models`
        # shim.
        import shrec.models.recurrence_manifold as RM
        monkeypatch.setattr(RM, "data_to_connectivity2", lambda X, **kw: A)

        # `_make_embedding` is still called on the input, so pass an X with
        # the right shape (n_timepoints, n_responses); contents are irrelevant.
        dummy_X = np.zeros((A.shape[0], 1))

        model = RecurrenceManifold(n_components=1, standardize=False)
        model.fit(dummy_X)
        pred = np.asarray(model.labels_).squeeze()
        pred = pred / (np.linalg.norm(pred) + 1e-12)

        # Cosine similarity, not Pearson correlation: do NOT subtract the mean.
        # The fixed (Fiedler) output is zero-mean by construction; the current
        # (SVD) output is a one-sided block indicator with non-zero mean.
        # Subtracting the mean would collapse any block-piecewise-constant
        # vector onto the analytical Fiedler direction, masking the bug.
        cos = abs(float(np.dot(pred, fiedler)))
        # Expected today: |cos| ≈ 0.866 (TruncatedSVD's 2nd singular vector
        # is the indicator of the smaller block, not the block-difference
        # Fiedler vector). After the §6 step-2 fix this should be ≈ 1.
        assert cos > 0.99, (
            f"|cos(pred, analytical Fiedler)| = {cos:.4f}; "
            "RecurrenceManifold is not returning the Laplacian Fiedler vector."
        )


def _degree_heterogeneous_affinity(p=20, w_low=1.0, w_high=8.0, bridge=0.5):
    """Two equal-size blocks with very different internal edge weights, so the
    high-weight block's nodes have ~`w_high/w_low`× the degree. The community
    structure (block 1 vs block 2) is unambiguous, but the *degree* imbalance
    is what distinguishes RatioCut (unnormalised) from NCut (normalised)."""
    N = 2 * p
    A = np.zeros((N, N))
    A[:p, :p] = w_low
    A[p:, p:] = w_high
    A[p - 1, p] = A[p, p - 1] = bridge
    np.fill_diagonal(A, 0.0)
    return A


def _fiedler_from_affinity(A, **model_kwargs):
    """Run RecurrenceManifold's spectral step on a precomputed affinity."""
    import shrec.models.recurrence_manifold as RM
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RM, "data_to_connectivity2", lambda X, **kw: A)
        dummy_X = np.zeros((A.shape[0], 1))
        model = RecurrenceManifold(standardize=False, **model_kwargs).fit(dummy_X)
    return np.asarray(model.labels_)


class TestNormalizedLaplacianOption:
    """`normalize_laplacian=True` switches the spectral step to the NCut /
    random-walk generalised eigenproblem `L v = λ D v` (the paper's
    "preconditioning" remedy for response bias). It must still recover an
    unambiguous block structure, and must produce a *measurably different*
    embedding from the unnormalised default under degree heterogeneity (proof
    the flag actually changes the operator).
    """

    def test_normalized_recovers_clean_block_split(self):
        from sklearn.metrics import adjusted_rand_score
        A = _two_block_affinity(p1=20, p2=20, bridge=1e-3)
        v = _fiedler_from_affinity(A, n_components=1, normalize_laplacian=True)
        split = (v > np.median(v)).astype(int)
        truth = np.array([0] * 20 + [1] * 20)
        assert adjusted_rand_score(truth, split) == 1.0

    def test_normalized_differs_from_unnormalized_under_degree_bias(self):
        A = _degree_heterogeneous_affinity()
        v_unnorm = _fiedler_from_affinity(A, normalize_laplacian=False)
        v_norm = _fiedler_from_affinity(A, normalize_laplacian=True)
        v_unnorm = v_unnorm / (np.linalg.norm(v_unnorm) + 1e-12)
        v_norm = v_norm / (np.linalg.norm(v_norm) + 1e-12)
        cos = abs(float(np.dot(v_unnorm, v_norm)))
        assert cos < 0.95, (
            f"|cos| = {cos:.3f}: normalize_laplacian had no measurable effect; "
            "the generalised eigenproblem is probably not being used."
        )


class TestEigenvectorShapeContract:
    """MM-contract — `subset_by_index=[1, n_components]` returns exactly
    `n_components` non-trivial eigenvectors (indices 1..n_components). Pins the
    output shape so a future off-by-one in the index range is caught.
    """

    @pytest.mark.parametrize("n_components,expected_ndim", [(1, 1), (2, 2), (3, 2)])
    def test_label_shape_matches_n_components(self, n_components, expected_ndim):
        A = _two_block_affinity(p1=15, p2=15, bridge=1e-2)
        v = _fiedler_from_affinity(A, n_components=n_components)
        assert v.ndim == expected_ndim
        if n_components == 1:
            assert v.shape == (A.shape[0],)
        else:
            assert v.shape == (A.shape[0], n_components)


class TestConnectivityGuard:
    """When the consensus graph is (nearly) disconnected, λ₂ ≈ 0 and the
    Fiedler eigenvector degenerates into a connected-component indicator
    rather than a smooth driver coordinate. `RecurrenceManifold.fit` must
    warn so the silent failure becomes visible. A well-connected graph (the
    MM22 bridge=1e-3 case) must NOT warn.
    """

    def _fit_on_affinity(self, A):
        import shrec.models.recurrence_manifold as RM
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(RM, "data_to_connectivity2", lambda X, **kw: A)
            dummy_X = np.zeros((A.shape[0], 1))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                RecurrenceManifold(n_components=1, standardize=False).fit(dummy_X)
        return [w for w in caught if "disconnected" in str(w.message)]

    def test_disconnected_graph_warns(self):
        A = _two_block_affinity(p1=10, p2=30, bridge=0.0)  # no bridge at all
        assert self._fit_on_affinity(A), (
            "RecurrenceManifold silently returned a component indicator on a "
            "disconnected graph; expected a connectivity warning."
        )

    def test_connected_graph_does_not_warn(self):
        A = _two_block_affinity(p1=10, p2=30, bridge=1e-3)  # MM22 case
        assert not self._fit_on_affinity(A), (
            "Connectivity warning fired on a well-connected graph (the MM22 "
            "oracle case); the λ₂ threshold is too aggressive."
        )


def _cycle_affinity(n):
    """Unweighted ring graph C_n: A[i,j] = 1 iff i and j are cyclic neighbours."""
    A = np.zeros((n, n))
    i = np.arange(n)
    A[i, (i + 1) % n] = 1.0
    A[i, (i - 1) % n] = 1.0
    return A


def _harmonic_basis(n, m):
    """Orthonormal {cos, sin} pair for the m-th Fourier harmonic on C_n."""
    i = np.arange(n)
    c = np.cos(2 * np.pi * m * i / n)
    s = np.sin(2 * np.pi * m * i / n)
    return np.column_stack([c / np.linalg.norm(c), s / np.linalg.norm(s)])


class TestCycleGraphFiedlerOracle:
    """§5b.3 MM23 — the Laplacian of the ring graph C_n is circulant, so its
    eigenvectors are the Fourier modes `cos(2π m i/n)`, `sin(2π m i/n)` with
    eigenvalues `2 − 2cos(2π m/n)`. The Fiedler eigenvalue (m=1) is **doubly
    degenerate**, so the Fiedler vector is determined only up to sign *and*
    phase — it is any unit vector in the first-harmonic plane span{cos, sin}.
    The precise, phase-robust oracle is therefore a *subspace* claim, not
    "the answer is cos": the recovered eigenvector(s) must lie in that plane,
    and at the first harmonic rather than a higher one.
    """

    def test_fiedler_pair_spans_first_harmonic(self):
        n = 60
        A = _cycle_affinity(n)
        V = np.atleast_2d(_fiedler_from_affinity(A, n_components=2))
        if V.shape[0] != n:  # normalise to (n, 2)
            V = V.T
        H1 = _harmonic_basis(n, 1)
        P1 = H1 @ H1.T  # projector onto the first-harmonic plane
        for col in V.T:
            captured = np.linalg.norm(P1 @ col) / (np.linalg.norm(col) + 1e-12)
            assert captured > 0.999, (
                f"recovered Fiedler vector leaves {1 - captured:.2e} of its norm "
                "outside the first-harmonic plane — not a pure C_n Fiedler mode."
            )

    def test_fiedler_is_first_harmonic_not_higher(self):
        n = 60
        A = _cycle_affinity(n)
        v = np.asarray(_fiedler_from_affinity(A, n_components=1)).squeeze()
        v = v / (np.linalg.norm(v) + 1e-12)
        P1 = _harmonic_basis(n, 1) @ _harmonic_basis(n, 1).T
        P2 = _harmonic_basis(n, 2) @ _harmonic_basis(n, 2).T
        # Almost all energy in the first harmonic, almost none in the second.
        assert np.linalg.norm(P1 @ v) > 0.999
        assert np.linalg.norm(P2 @ v) < 0.05
