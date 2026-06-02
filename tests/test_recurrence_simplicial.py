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
    constructs them: self and true duplicates dropped via the 1e-10 → inf floor,
    then the k nearest distinct neighbours (`[:k]`, so `dists[0]` is the nearest
    — the ρ convention of Appendix B; cf. the MM5 off-by-one fix)."""
    dmat = cdist(X, X)
    dmat[dmat < 1e-10] = np.inf
    dists = np.partition(dmat, k, axis=1)
    return np.sort(dists, axis=1)[:, :k]


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
    def test_sigma_solve_is_a_valid_root_or_fallback(self, X):
        # Mirror the canonical neighbour construction (self/dups -> inf, then
        # the k nearest; cf. the MM5 ρ off-by-one fix).
        dmat = cdist(X, X)
        dmat[dmat < 1e-10] = np.inf
        dists = np.sort(np.partition(dmat, _K, axis=1), axis=1)[:, :_K]
        target = np.log2(_K)
        for row in dists:
            rho, sigma = fit_rho_sigma(row, _K)
            assert sigma > 0 and np.isfinite(sigma)
            residual = abs(np.sum(np.exp(-relu(row - rho) / sigma)) - target)
            tied_fallback = np.isclose(sigma, rho)
            # `brentq` locates the σ-*root* to its tolerance (xtol), not the
            # function *residual* — which is ≈ f'(σ*)·xtol. In a near-tied
            # neighbourhood the root σ is tiny and f's slope ~1/σ is huge, so a
            # correctly-bracketed root can leave a residual that grows like
            # xtol/ε (observed up to ~1e-5 on adversarial Hypothesis inputs).
            # That is still a *good* solve, qualitatively unlike the fsolve
            # stall it replaced (σ stuck at ρ, residual ≈ 2.8). So this
            # robustness test only asserts the solver isn't catastrophically
            # wrong (1e-3 separates a valid root from a stall by 3+ orders of
            # magnitude); the tight residual < 1e-6 bound on well-conditioned
            # input is MM1 (`TestDefiningEquation`). See Round 7/11 of
            # docs/math-learning-notes.md.
            assert residual < 1e-3 or tied_fallback, (
                f"row solve neither satisfied the equation (residual="
                f"{residual:.3g}) nor took the ρ-fallback (σ={sigma:.3g}, "
                f"ρ={rho:.3g})."
            )


# --- §5b.1 MM5 — cross-implementation parity --------------------------------

class TestSimplicialParity:
    """MM5 — reconcile `fit_rho_sigma` (our self-contained σ-solver, the
    canonical pipeline since the refactor dropped the umap-learn dependency)
    against `umap.umap_.fuzzy_simplicial_set` / `smooth_knn_dist`.

    The historical strict-xfail recorded "they diverge on σ-solver
    conventions." Pinned down (see docs/math-learning-notes.md Round 11), the
    divergence was exactly two things:

      1. A real **off-by-one bug** in `dataset_to_simplex`: it inf-filled the
         self-distance *and then* sliced `[1:k+1]`, double-skipping so ρ became
         the *second*-nearest neighbour. Appendix B (and umap) use the
         *nearest*. Fixed to `[:k]`; ρ now matches umap to floating point.
      2. A genuine, intentional **convention** difference in what "k" counts:
         umap's `n_neighbors = k` *includes the query point itself*, so it sums
         over k-1 real neighbours toward a target of log₂(k). Our paper-faithful
         convention uses k real neighbours toward log₂(k). Adopt umap's
         self-counting (feed the k-1 nearest) and σ matches umap to ~1e-6.

    So the two now agree exactly once the self-counting convention is matched;
    the only remaining difference is that documented convention choice.
    """

    def test_rho_matches_umap_exactly(self, rng):
        """After the off-by-one fix, ρ_i (nearest-neighbour distance) is
        identical to umap's `rhos` (its `local_connectivity=1` ρ)."""
        umap_mod = pytest.importorskip("umap.umap_")
        X = rng.standard_normal((50, 3))
        k = 10

        _, _, rhos_umap, _ = umap_mod.fuzzy_simplicial_set(
            X, k, 0, "euclidean", return_dists=True,
        )
        rhos_umap = np.asarray(rhos_umap)

        knn = _knn_dists(X, k)
        rhos_ours = np.array([fit_rho_sigma(row, k)[0] for row in knn])
        np.testing.assert_allclose(rhos_ours, rhos_umap, atol=1e-6)

    def test_sigma_matches_umap_under_self_counting_convention(self, rng):
        """umap's `n_neighbors` counts the self-point, so it sums over k-1 real
        neighbours toward log₂(k). Match that convention and σ agrees to ~1e-6
        — proving the σ-solvers are otherwise identical."""
        umap_mod = pytest.importorskip("umap.umap_")
        X = rng.standard_normal((50, 3))
        k = 10

        _, sigmas_umap, _, _ = umap_mod.fuzzy_simplicial_set(
            X, k, 0, "euclidean", return_dists=True,
        )
        sigmas_umap = np.asarray(sigmas_umap)

        knn = _knn_dists(X, k)
        # Feed the k-1 nearest with target log₂(k): umap's self-counting.
        sigmas_ours = np.array([fit_rho_sigma(row[: k - 1], k)[1] for row in knn])
        np.testing.assert_allclose(sigmas_ours, sigmas_umap, atol=1e-5)
