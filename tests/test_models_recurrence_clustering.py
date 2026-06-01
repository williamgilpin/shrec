"""Math-correctness tests for `RecurrenceClustering` (discrete driver).

Tied to docs/tests-math.md §5b.3 — **MM19** (period-2 Sauer limit) and
**MM20** (period-4) are two of the four "must" tests called out for the
canonical-algorithm PR (§6 step 2).

Both make a falsifiable claim from the paper (§II.B line 461; §III.D
line 570): in the noise-free Sauer limit, SHREC recovers the driver's
equivalence classes (`adjusted_rand_score → 1.0` as `N, T → ∞`).

Post §6-step-2: the canonical simplicial-complex + Laplacian-Fiedler
refactor is in place. At finite N=20 / T=1000, Leiden community detection
at the default `resolution=1.0` slightly over-segments period-2 (ARI ≈
0.99) and *collapses* period-4 to two communities (ARI ≈ 0.50). The
period-2 case still clears a near-exact threshold; the period-4 case is
marked xfail and diagnosed as a **representational** limit (NOT a Leiden-
resolution artifact — an oracle spectral k-means=4 and the continuous
manifold both fail too; see `test_period_four_is_not_separable_in_graph`).
MM21 (period-8) confirms the limit *worsens* with more driver levels:
recovery sits in a partial-collapse band well below the naive >0.85 target.
"""
import numpy as np
import pytest

graspologic = pytest.importorskip("graspologic")

from sklearn.metrics import adjusted_rand_score

from shrec.models.models import RecurrenceClustering


def _logistic_ensemble(driver, n_responses=20, coupling=0.5, seed=0):
    """Drive `n_responses` distinct chaotic logistic maps with `driver`.

    Mirrors the construction in `tests/conftest.py:square_wave_responses`,
    generalised to an arbitrary driver array. Returns shape (T, n_responses).
    """
    rng = np.random.default_rng(seed)
    T = len(driver)
    r_values = rng.uniform(3.81, 3.97, size=n_responses)
    responses = np.empty((T, n_responses))
    for k, r in enumerate(r_values):
        x = np.empty(T)
        x[0] = rng.uniform(0.1, 0.9)
        for t in range(T - 1):
            x[t + 1] = r * x[t] * (1 - x[t]) + coupling * driver[t]
            x[t + 1] = np.clip(x[t + 1], 0.0, 1.0)
        responses[:, k] = x
    return responses


class TestSauerLimitNearExactRecovery:
    """The paper claims exact equivalence-class recovery in the noise-free
    Sauer limit. These tests pin a *near-exact* version of that claim.
    """

    def test_period_two_driver_ari_near_one(self):
        """MM19 — period-2 driver, N=20 responses, zero noise → ARI ≈ 1."""
        T = 1000
        z = np.where(np.arange(T) % 2 == 0, 0.2, 0.8)
        X = _logistic_ensemble(z, n_responses=20, coupling=0.5, seed=0)

        # graspologic's leiden requires a *positive* random_seed (or None).
        model = RecurrenceClustering(random_state=1)
        model.fit(X)

        ari = adjusted_rand_score(z, model.labels_)
        assert ari > 0.95, (
            f"Sauer-limit period-2 recovery: ARI = {ari:.4f} (expected > 0.95). "
            "RecurrenceClustering is not on the canonical simplicial pipeline."
        )

    @pytest.mark.xfail(
        reason=(
            "Period-4 driver yields ARI ≈ 0.50 instead of the paper's "
            "asymptotic ARI = 1. NOT a Leiden-resolution artifact: diagnosed "
            "as representational (see test_period_four_is_not_separable_in_graph "
            "below). A modularity-resolution sweep jumps straight from 2 "
            "communities (res≤1) to ~1000 singletons (res≥2) with no stable "
            "4-community regime; an *oracle* spectral k-means=4 on the consensus "
            "Laplacian also only reaches ARI≈0.5 across N∈[20,100] and "
            "coupling∈[0.5,1]; and the continuous RecurrenceManifold reaches "
            "Spearman |ρ|≈0.38. The four levels {0.1,0.4,0.6,0.9} collapse into "
            "a low/high 2-way split in the recurrence graph, so closing MM20 "
            "needs a richer recurrence representation, not a clustering tweak."
        ),
        strict=True,
    )
    def test_period_four_driver_ari_near_one(self):
        """MM20 — period-4 driver, N=20 responses, zero noise → ARI ≈ 1."""
        T = 1000
        levels = np.array([0.1, 0.4, 0.6, 0.9])
        z = levels[np.arange(T) % 4]
        X = _logistic_ensemble(z, n_responses=20, coupling=0.5, seed=0)

        model = RecurrenceClustering(random_state=1)
        model.fit(X)

        ari = adjusted_rand_score(z, model.labels_)
        assert ari > 0.95, (
            f"Sauer-limit period-4 recovery: ARI = {ari:.4f} (expected > 0.95)."
        )

    def test_period_four_is_not_separable_in_graph(self):
        """MM20 diagnosis (characterisation test) — pins *why* MM20 is xfail:
        the period-4 structure is absent from the consensus graph, so even an
        oracle spectral k-means into 4 clusters cannot recover it. If a future
        change to the recurrence representation makes period-4 separable, this
        test will start failing and flag that MM20's xfail should be revisited.
        """
        import scipy.linalg
        from sklearn.cluster import KMeans

        from shrec.recurrence import data_to_connectivity2

        T = 1000
        levels = np.array([0.1, 0.4, 0.6, 0.9])
        z = levels[np.arange(T) % 4]
        X = _logistic_ensemble(z, n_responses=20, coupling=0.5, seed=0)

        model = RecurrenceClustering(random_state=1)
        embedded = model._make_embedding(model._preprocess(X))
        affinity = data_to_connectivity2(embedded, time_exclude=0)
        laplacian = np.diag(affinity.sum(1)) - affinity
        _, eigvecs = scipy.linalg.eigh(laplacian, subset_by_index=[1, 3])
        oracle = KMeans(4, n_init=10, random_state=0).fit_predict(eigvecs)

        ari = adjusted_rand_score(z, oracle)
        assert ari < 0.6, (
            f"Oracle spectral k-means=4 recovered period-4 with ARI={ari:.3f} "
            "(≥ 0.6): the consensus graph now separates the four levels, so the "
            "MM20 limitation may be representational no longer — revisit the xfail."
        )


class TestPeriodEightRepresentationalLimit:
    """MM21 (§5b.3, characterisation) — the catalog optimistically targeted
    ARI > 0.85 for a period-8 stochastically-forced driver, with the note
    "use the continuous path if discrete collapses." Both collapse: this is
    the SAME representational limit as MM20 (period-4), and it *worsens* with
    more driver levels. We pin the partial-collapse band — clearly below the
    >0.85 target (so the limit is real) but clearly above chance (so structure
    is partially present, ruling out a pure bug). If a future recurrence
    representation lifts either number past 0.85, this test fires and flags
    that the period-8 limitation, like MM20, may be representational no longer.
    """

    @pytest.mark.slow
    def test_period_eight_recovery_is_partial(self):
        from scipy.stats import spearmanr

        from shrec.models.models import RecurrenceManifold

        T = 800
        levels = np.linspace(0.1, 0.9, 8)
        labels_true = np.arange(T) % 8
        z = levels[labels_true]
        rng = np.random.default_rng(0)
        z_forced = z + 0.04 * rng.standard_normal(T)  # stochastic forcing
        X = _logistic_ensemble(z_forced, n_responses=20, coupling=0.5, seed=0)

        ari = adjusted_rand_score(
            labels_true, RecurrenceClustering(random_state=1).fit(X).labels_
        )
        v = RecurrenceManifold(random_state=1).fit(X).labels_
        rho = abs(spearmanr(v, z).correlation)

        assert 0.25 < ari < 0.85, (
            f"period-8 discrete ARI={ari:.3f} left the partial-collapse band "
            "(0.25, 0.85); the representational limit may have changed — "
            "revisit MM21/MM20."
        )
        assert rho < 0.85, (
            f"period-8 continuous |ρ|={rho:.3f} ≥ 0.85: the manifold now "
            "resolves 8 levels — the MM20/MM21 limit may be representational "
            "no longer."
        )
