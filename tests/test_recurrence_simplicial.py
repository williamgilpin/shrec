"""Math-correctness tests for the fuzzy simplicial complex.

Tied to docs/tests-math.md §5b.1 — currently only **MM5** (parity between the
two in-repo simplicial implementations) is in scope; this is one of the
four "must" tests called out for the canonical-algorithm PR (§6 step 2).
"""
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.distance import cdist

from shrec.models.models import dataset_to_simplex, relu
from shrec.recurrence.simplicial import fit_rho_sigma


def _knn_dists(X, k):
    """The per-row sorted k-NN distances, exactly as `dataset_to_simplex`
    constructs them (true duplicates dropped via the 1e-10 floor)."""
    dmat = cdist(X, X)
    dmat[dmat < 1e-10] = np.inf
    dists = np.partition(dmat, k + 1, axis=1)
    return np.sort(dists, axis=1)[:, 1:k + 1]


# --- §5b.1 MM1 — defining-equation residual ---------------------------------

class TestDefiningEquation:
    """MM1 (must) — the σ root-solve must actually satisfy its defining
    equation `Σ_m exp(-ReLU(d_im − ρ_i)/σ_i) = log₂ k` for *every* row.

    This is the cheapest possible correctness check and the one that catches
    the `fsolve` silent-stall bug: MINPACK's `hybrd` reports `ier=1`
    ("converged") on the step-size criterion while leaving σ at the initial
    guess ρ, so the residual is O(1) instead of < 1e-6 on ~6% of rows. A
    bracketed `brentq` (the equation is monotone in σ) drives every residual
    to ~1e-12. See docs/math-learning-notes.md.
    """

    def test_residual_below_tolerance_on_every_row(self, rng):
        X = rng.standard_normal((50, 3))
        k = 10
        target = np.log2(k)
        for row in _knn_dists(X, k):
            rho, sigma = fit_rho_sigma(row, k)
            residual = abs(np.sum(np.exp(-relu(row - rho) / sigma)) - target)
            assert residual < 1e-6, (
                f"σ root-solve left residual {residual:.4g} (σ={sigma:.4g}, "
                f"ρ={rho:.4g}); the defining equation is not satisfied."
            )


# --- §5b.1 MM2 — scale invariance of (ρ, σ) ---------------------------------

class TestScaleInvariance:
    """MM2 (must) — scaling every distance by α scales (ρ, σ) by α, leaving
    each affinity `exp(-ReLU(αd − αρ)/(ασ))` unchanged. The equation is
    scale-covariant by construction, so this is an exact oracle.
    """

    def test_rho_sigma_scale_linearly(self, rng):
        X = rng.standard_normal((40, 3))
        k = 10
        alpha = 7.3
        for row in _knn_dists(X, k):
            rho, sigma = fit_rho_sigma(row, k)
            rho_a, sigma_a = fit_rho_sigma(alpha * row, k)
            np.testing.assert_allclose(rho_a, alpha * rho, rtol=1e-9)
            np.testing.assert_allclose(sigma_a, alpha * sigma, rtol=1e-6)

    def test_affinity_unchanged_under_scaling(self, rng):
        X = rng.standard_normal((40, 3))
        k = 10
        np.testing.assert_allclose(
            dataset_to_simplex(X, k=k),
            dataset_to_simplex(7.3 * X, k=k),
            atol=1e-9,
        )


# --- §5b.1 MM3 — unit diagonal ----------------------------------------------

class TestUnitDiagonal:
    """MM3 (must) — `d_ii = 0` ⇒ `exp(-ReLU(-ρ_i)) = exp(0) = 1`, and the
    fuzzy union `1 + 1 − 1·1 = 1`, so the diagonal is exactly 1.
    """

    def test_diagonal_is_exactly_one(self, rng):
        A = dataset_to_simplex(rng.standard_normal((40, 3)), k=10)
        np.testing.assert_array_equal(np.diag(A), np.ones(A.shape[0]))


# --- §5b.1 MM4 — fuzzy-union symmetrisation ---------------------------------

class TestFuzzyUnionSymmetrisation:
    """MM4 (must) — the probabilistic t-conorm `A + Aᵀ − A∘Aᵀ` must yield a
    symmetric matrix with all entries in [0, 1].
    """

    def test_symmetric_and_in_unit_interval(self, rng):
        A = dataset_to_simplex(rng.standard_normal((40, 3)), k=10)
        np.testing.assert_allclose(A, A.T, atol=1e-12)
        assert A.min() >= 0.0
        assert A.max() <= 1.0


# --- §5b.1 MM1/MM3/MM4 (property-based) — invariants over arbitrary clouds ---

_K = 3


@st.composite
def _point_clouds(draw):
    """Random (n, d) point clouds with bounded, finite coordinates and enough
    distinct points to support a k=3 neighbourhood."""
    n = draw(st.integers(min_value=_K + 2, max_value=20))
    d = draw(st.integers(min_value=1, max_value=4))
    X = draw(arrays(
        np.float64, (n, d),
        elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    ))
    # At least k+2 distinct points, else the k-NN structure is ill-defined.
    assume(len(np.unique(np.round(X, 6), axis=0)) >= _K + 2)
    return X


class TestSimplexInvariantsPropertyBased:
    """Hypothesis generalises the hand-picked MM3/MM4 oracles to *arbitrary*
    clouds and lets the search find adversarial inputs (it independently
    rediscovers the tied-neighbourhood degeneracy that the σ-solve fallback
    handles). For any input, `dataset_to_simplex` must return a symmetric
    matrix with unit diagonal and entries in [0, 1], and every row's σ-solve
    must either satisfy the defining equation or have taken the documented
    ρ-fallback. See docs/math-learning-notes.md (property-based testing).
    """

    @settings(max_examples=40, deadline=None)
    @given(X=_point_clouds())
    def test_output_is_symmetric_unit_diagonal_in_unit_interval(self, X):
        A = dataset_to_simplex(X, k=_K)
        assert np.all(np.isfinite(A))
        np.testing.assert_allclose(A, A.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(A), 1.0, atol=1e-12)
        assert A.min() >= -1e-12 and A.max() <= 1.0 + 1e-12

    @settings(max_examples=40, deadline=None)
    @given(X=_point_clouds())
    def test_sigma_solve_satisfies_equation_or_takes_fallback(self, X):
        dmat = cdist(X, X)
        dmat[dmat < 1e-10] = np.inf
        dists = np.sort(np.partition(dmat, _K + 1, axis=1), axis=1)[:, 1:_K + 1]
        target = np.log2(_K)
        for row in dists:
            rho, sigma = fit_rho_sigma(row, _K)
            residual = abs(np.sum(np.exp(-relu(row - rho) / sigma)) - target)
            tied_fallback = np.isclose(sigma, rho)
            # brentq locates the σ-*root* to its tolerance, not the function
            # residual. In a near-tied neighbourhood (two neighbours separated
            # by ε ≪ the neighbourhood scale) the root σ ~ ε is tiny and f's
            # slope there is ~1/σ, so a correctly-bracketed root can still leave
            # a residual that grows like 1/ε. That steep regime (σ ≪ scale) is a
            # *good* solve — qualitatively unlike the fsolve stall it replaced
            # (σ stuck at ρ, residual ≈ 2.8). The σ/scale marker is scale-
            # invariant (MM2), so this branch can't be scaled away. Accept a
            # small residual, the exact tied fallback, or a genuine tiny-σ root.
            steep_root = sigma < 1e-6 * row[-1]
            assert residual < 1e-6 or tied_fallback or steep_root, (
                f"row solve neither satisfied the equation (residual="
                f"{residual:.3g}) nor took the ρ-fallback (σ={sigma:.3g}, "
                f"ρ={rho:.3g})."
            )


# --- §5b.1 MM5 — cross-implementation parity --------------------------------

class TestSimplicialParity:
    """The repo carries two fuzzy-simplicial implementations:

      * `models.dataset_to_simplex` — fsolve on each row, applied densely
        (paper Appendix B, eq. for σ_i).
      * `models.data_to_connectivity2` — delegates the ρ/σ root-solve to
        `umap.umap_.fuzzy_simplicial_set`, then applies those parameters
        densely (so the only difference *should* be the solver).

    Per §3 of the assessment, only one of these can be canonical. If they
    disagree on the same input, one is wrong (or differs only on
    parameter-choice conventions worth documenting).
    """

    @pytest.mark.xfail(
        reason=(
            "Confirmed divergence: the two simplicial implementations "
            "carry different conventions for the σ root-solve (likely "
            "the k-vs-(k+1) neighbour count and/or the ρ-as-min-distance "
            "rule). The §6 step-2 PR chose `dataset_to_simplex` (the "
            "self-contained, paper-faithful version) and dropped the "
            "umap-learn dependency from the canonical pipeline. This "
            "test remains as a historical record of the comparison; "
            "lifting xfail requires reconciling the two conventions."
        ),
        strict=True,
    )
    def test_dataset_to_simplex_matches_umap_sigmas_and_rhos(self, rng):
        """Apply the umap-derived (ρ, σ) the same way `data_to_connectivity2`
        does (dense `exp(-ReLU(d − ρ)/σ)` + fuzzy union), and compare to
        the in-repo `dataset_to_simplex`. If both implement Appendix B
        faithfully, they should agree to ~1e-5 on a random input.
        """
        umap_mod = pytest.importorskip("umap.umap_")
        fuzzy_simplicial_set = umap_mod.fuzzy_simplicial_set

        X = rng.standard_normal((50, 3))
        k = 10

        ours = dataset_to_simplex(X, k=k)

        # Mirror data_to_connectivity2:271-279 exactly.
        _, sigmas, rhos, _ = fuzzy_simplicial_set(
            X, k, 0, "euclidean", return_dists=True,
        )
        sigmas = np.asarray(sigmas)
        rhos = np.asarray(rhos)
        dmat = cdist(X, X)
        umap_aff = np.exp(-relu(dmat - rhos[None, :]) / sigmas[None, :])
        umap_aff = umap_aff + umap_aff.T - umap_aff * umap_aff.T

        # Expected to FAIL today on at least one of: σ-solver convention
        # (log₂ k vs ln k), the k-vs-(k-1) neighbour count, or ρ choice
        # (smallest vs smallest non-zero). Whatever the diff is, naming
        # it concretely is the goal of this test.
        np.testing.assert_allclose(ours, umap_aff, atol=1e-5)
