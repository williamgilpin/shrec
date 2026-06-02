"""Paper-claim regression tests — quantitative scaling laws (docs/tests-math.md
§5b.4). Slow; run under `-m slow` / nightly CI.

MM27 — accuracy scaling with total data `NT/τ` (paper Appendix E.2):

    Acc(NT/τ) = Acc_max · (1 − exp(−β·√(NT/τ)))

i.e. reconstruction accuracy rises and saturates as more responses are added.
The paper states this for the discrete-driver ARI; we test it on the
continuous-driver path (RecurrenceManifold + Spearman |ρ|), which both (a)
sidesteps the period-4 Leiden-resolution collapse that caps the discrete ARI
(MM20, xfail) and (b) gives a smooth accuracy in [0, 1] that the saturating
form can be fit against. The scaling only becomes visible in a mildly
data-limited regime, so we add light observation noise (σ=0.05) — with none,
accuracy saturates at N=2 and there is nothing to fit. See
docs/math-learning-notes.md (Round 5).
"""
import numpy as np
import pytest
from scipy.optimize import curve_fit
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from shrec.models import RecurrenceClustering, RecurrenceManifold
from shrec.recurrence import data_to_connectivity2

T = 500
TAU = 1
N_SWEEP = np.array([2, 4, 8, 16, 32])
N_SEEDS = 8
OBS_NOISE = 0.05


def _logistic_ensemble(driver, n_responses, seed, obs_noise=OBS_NOISE):
    """`n_responses` chaotic logistic maps forced by `driver`, each observed
    through light additive noise. Distinct seed per (N, seed) cell."""
    rng = np.random.default_rng(1000 * seed + n_responses)
    n = len(driver)
    r_values = rng.uniform(3.7, 3.9, size=n_responses)
    X = np.empty((n, n_responses))
    for k, r in enumerate(r_values):
        x = np.empty(n)
        x[0] = rng.uniform(0.1, 0.9)
        for t in range(n - 1):
            x[t + 1] = np.clip(r * x[t] * (1 - x[t]) + 0.4 * driver[t], 0.0, 1.0)
        X[:, k] = x + obs_noise * rng.standard_normal(n)
    return X


@pytest.mark.slow
class TestAccuracyScalingLaw:
    """MM27 — fit the saturating-exponential accuracy law and assert its
    qualitative content: accuracy grows with total data (β > 0), saturates at
    a usable level (Acc_max), improves markedly from the smallest to the
    largest ensemble, and the saturating form fits better than a flat line.
    """

    def test_accuracy_saturates_with_ensemble_size(self):
        driver = 0.5 + 0.4 * np.sin(2 * np.pi * np.arange(T) / 100.0)

        mean_acc = np.empty(len(N_SWEEP))
        for i, n_resp in enumerate(N_SWEEP):
            accs = []
            for seed in range(N_SEEDS):
                X = _logistic_ensemble(driver, int(n_resp), seed)
                v = RecurrenceManifold(random_state=1).fit(X).labels_
                accs.append(abs(spearmanr(v, driver).correlation))
            mean_acc[i] = np.mean(accs)

        # Fit Acc(x) = Acc_max (1 − exp(−β x)) with x = √(NT/τ).
        x = np.sqrt(N_SWEEP * T / TAU)

        def law(x, acc_max, beta):
            return acc_max * (1.0 - np.exp(-beta * x))

        (acc_max, beta), _ = curve_fit(
            law, x, mean_acc, p0=[0.8, 1e-3],
            bounds=([0.0, 0.0], [1.0, 1.0]), maxfev=10000,
        )

        # Goodness of fit vs a flat-mean baseline.
        resid = mean_acc - law(x, acc_max, beta)
        r2 = 1.0 - np.sum(resid ** 2) / np.sum((mean_acc - mean_acc.mean()) ** 2)

        assert beta > 0.0, "accuracy did not increase with total data NT/τ"
        assert 0.6 <= acc_max <= 1.0, f"saturation accuracy {acc_max:.2f} off"
        assert mean_acc[-1] - mean_acc[0] > 0.25, (
            f"no clear accuracy gain from N={N_SWEEP[0]} to N={N_SWEEP[-1]}: "
            f"{mean_acc[0]:.2f} → {mean_acc[-1]:.2f}"
        )
        assert r2 > 0.5, (
            f"saturating-exponential form did not fit (R²={r2:.2f}); "
            f"means={np.round(mean_acc, 3)}"
        )
        # Near-monotone in the means (allow small sampling dips).
        assert np.all(np.diff(mean_acc) > -0.05)


# MM28 — percolation order parameter (Appendix E.3).
PERC_T = 300
PERC_N_SWEEP = [2, 4, 8, 16, 32]
PERC_SEEDS = 10
PERC_COUPLING = 0.1   # weak: single responses are individually ambiguous
PERC_THETA = 0.55     # absolute edge threshold on the consensus affinity


def _weakly_coupled_period2(n_responses, seed):
    """Period-2 driver, weakly coupled into chaotic logistic responses. Weak
    coupling is essential: each response *alone* gives a near-percolated
    recurrence graph, so the consensus only resolves the two driver-state
    basins once enough responses are averaged — which is the regime in which
    the percolation transition is visible as N grows."""
    rng = np.random.default_rng(seed)
    z = np.where(np.arange(PERC_T) % 2 == 0, 0.2, 0.8)
    r_values = rng.uniform(3.81, 3.97, size=n_responses)
    X = np.empty((PERC_T, n_responses))
    for k, r in enumerate(r_values):
        x = np.empty(PERC_T)
        x[0] = rng.uniform(0.1, 0.9)
        for t in range(PERC_T - 1):
            x[t + 1] = np.clip(r * x[t] * (1 - x[t]) + PERC_COUPLING * z[t], 0.0, 1.0)
        X[:, k] = x
    return X


def _largest_cc_fraction(affinity, theta):
    """T_LCC/T: fraction of nodes in the largest connected component of the
    consensus graph binarised at an *absolute* edge threshold θ."""
    adj = affinity > theta
    np.fill_diagonal(adj, False)
    _, labels = connected_components(adj.astype(int), directed=False)
    return np.bincount(labels).max() / affinity.shape[0]


@pytest.mark.slow
class TestPercolationOrderParameter:
    """MM28 (§5b.4, Appendix E.3) — the scaled largest connected component
    `T_LCC/T ∈ [0,1]` of the aggregated consensus adjacency A is a percolation
    order parameter: it falls toward driver-state fragmentation as the amount
    of data grows ("percolation loss precedes accurate reconstruction", Fig 6).
    We reproduce this in N (number of responses).

    Two construction choices, both load-bearing and both lessons from the
    investigation (see docs/math-learning-notes.md Round 10/12):
      - **absolute** edge threshold, not a quantile: the consensus weight
        distribution shifts with N, so a *relative* threshold re-percolates the
        graph and the trend reverses.
      - **weak coupling**: with strong coupling every response imprints the same
        2-state structure, so the consensus is N-invariant and there is no
        transition to see; the transition lives where single responses are
        ambiguous and consensus does the work.
    Near criticality the order parameter is seed-noisy (the paper averages 60
    replicates), so we assert on the seed-averaged curve: a strong monotone-
    decreasing trend (Spearman) and a clear net drop.
    """

    def test_lcc_fraction_decreases_with_ensemble_size(self):
        mean_lcc = np.empty(len(PERC_N_SWEEP))
        for i, n_resp in enumerate(PERC_N_SWEEP):
            fracs = []
            for seed in range(PERC_SEEDS):
                X = _weakly_coupled_period2(n_resp, seed)
                model = RecurrenceClustering(random_state=1)
                embedded = model._make_embedding(model._preprocess(X))
                A = data_to_connectivity2(embedded, time_exclude=0)
                fracs.append(_largest_cc_fraction(A, PERC_THETA))
            mean_lcc[i] = np.mean(fracs)

        drop = mean_lcc[0] - mean_lcc[-1]
        rho = spearmanr(PERC_N_SWEEP, mean_lcc).correlation

        assert rho < -0.8, (
            f"percolation order parameter is not monotone-decreasing in N "
            f"(Spearman={rho:.2f}); LCC/T={np.round(mean_lcc, 3)}"
        )
        assert drop > 0.3, (
            f"no clear percolation loss from N={PERC_N_SWEEP[0]} to "
            f"N={PERC_N_SWEEP[-1]}: {mean_lcc[0]:.2f} → {mean_lcc[-1]:.2f}"
        )
        # No appreciable upward excursion (allow tiny criticality noise).
        assert np.all(np.diff(mean_lcc) <= 0.05)
