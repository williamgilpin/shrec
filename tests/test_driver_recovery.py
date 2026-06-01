"""End-to-end driver-recovery oracles (docs/tests-math.md §5b.3 MM24-MM26).

These are the "does the science actually work" tests: construct a known hidden
driver `z(t)`, observe it through an explicit measurement function, and check
that `RecurrenceManifold` reconstructs `z` up to a monotone transform (Spearman
|ρ|, which is invariant to any monotone reparametrisation — exactly the
indeterminacy SHREC's recovered coordinate carries).

The progression isolates the measurement model:
  - MM24 identity        x_k = z              (direct, noisy, observation)
  - MM25 linear          x_k = a_k z + b_k    (per-sensor gain/offset)
  - MM26 monotone        x_k = tanh(z/σ_k)    (saturating nonlinearity)
All three are *monotone* in z, so all three should recover it: the recurrence
manifold of a set of monotone observations of the same driver collapses onto a
single 1-D coordinate that tracks the driver's value.

The driver is a smooth, aperiodic signal (Gaussian-smoothed noise) rather than a
periodic one on purpose: a periodic driver revisits every value twice per cycle,
so the recurrence manifold recovers *phase* (bijective with time) instead of
*value*, and Spearman-vs-value would be artificially low. A slowly varying
aperiodic driver makes value and manifold-coordinate monotone-related, which is
what these oracles assert. (The periodic-driver / phase story is its own topic in
docs/math-learning-notes.md, Round 9.)
"""
import numpy as np
import pytest
from scipy.stats import spearmanr

from shrec.models import RecurrenceManifold

T = 600
N_RESPONSES = 6


def _smooth_driver(T, seed, scale=25.0):
    """A smooth, aperiodic, zero-mean unit-variance driver: white noise low-pass
    filtered with a Gaussian kernel of width `scale`. Slowly varying relative to
    the delay embedding, so its value (not just its phase) is recoverable."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(T)
    grid = np.arange(-3 * scale, 3 * scale + 1)
    kernel = np.exp(-0.5 * (grid / scale) ** 2)
    kernel /= kernel.sum()
    z = np.convolve(w, kernel, mode="same")
    return (z - z.mean()) / z.std()


def _recover(z, measurement, seed, obs_noise=0.02):
    """Observe `z` through `measurement(z, rng) -> x_k` on N channels, then run
    the continuous SHREC manifold and return |Spearman(recovered, z)|."""
    rng = np.random.default_rng(1000 + seed)
    X = np.empty((len(z), N_RESPONSES))
    for k in range(N_RESPONSES):
        X[:, k] = measurement(z, rng) + obs_noise * rng.standard_normal(len(z))
    v = RecurrenceManifold(random_state=1).fit(X).labels_
    return abs(spearmanr(v, z).correlation)


class TestDriverRecoveryUnderMeasurement:

    @pytest.mark.parametrize("seed", range(3))
    def test_identity_measurement(self, seed):
        """MM24 — direct (identity) observation of the driver: |ρ| > 0.95.

        Stated on N>1 noisy copies rather than the literal N=1 of the catalog,
        because `standardize_ts` squeezes an (T, 1) input to 1-D (cf. MM16)."""
        z = _smooth_driver(T, seed)
        rho = _recover(z, lambda z, rng: z.copy(), seed)
        assert rho > 0.95, f"identity-measurement recovery |ρ|={rho:.3f} ≤ 0.95"

    @pytest.mark.parametrize("seed", range(3))
    def test_linear_measurement(self, seed):
        """MM25 — per-sensor affine measurement x_k = a_k z + b_k: |ρ| > 0.9."""
        z = _smooth_driver(T, seed)
        rho = _recover(
            z,
            lambda z, rng: rng.uniform(0.5, 3.0) * z + rng.uniform(-2.0, 2.0),
            seed,
        )
        assert rho > 0.9, f"linear-measurement recovery |ρ|={rho:.3f} ≤ 0.9"

    @pytest.mark.parametrize("seed", range(3))
    def test_monotone_nonlinear_measurement(self, seed):
        """MM26 — saturating monotone measurement x_k = tanh(z/σ_k): |ρ| > 0.8."""
        z = _smooth_driver(T, seed)
        rho = _recover(z, lambda z, rng: np.tanh(z / rng.uniform(0.5, 2.0)), seed)
        assert rho > 0.8, f"tanh-measurement recovery |ρ|={rho:.3f} ≤ 0.8"
