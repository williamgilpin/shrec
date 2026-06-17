"""Tests for the composable pipeline (docs/decisions/0008, plan 0009).

Covers the new public surface the stage-strategy refactor introduced:

- **MM37 (must)** — preset ≡ `ShrecPipeline` equivalence. The four named
  models must be faithful thin presets over the composable spine.
- **MM40** — `Reconstructor` / `Connectivity` strategy contracts (output
  shapes, derived attributes, injection seam).
- **MM41 (slow)** — *off-preset* combinations recover a known driver, so the
  flexibility is real and not just an API that type-checks.

A note on which combinations are *sensible* (recorded in 0009): every
reconstructor carries an implicit input-semantics contract. `Fiedler` and
`Leiden` consume an affinity (high = connected), so any `Connectivity` feeds
them. `UnionFind` needs a *sparse/binary* adjacency (a dense affinity makes
every node mutually reachable → one component), so it pairs with
`ExpKernelConnectivity(sparsify=True)`. `Isomap` wants a dissimilarity. The
genuinely-novel *recovery* combos are therefore `ExpKernel × {Fiedler,
Leiden}` (swap the connectivity, keep a spectral/community reconstructor),
which is what MM41 exercises.
"""
import warnings

import numpy as np
import pytest

graspologic = pytest.importorskip("graspologic")

from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from shrec.models import (
    ClassicalRecurrenceClustering,
    HirataNomuraIsomap,
    RecurrenceClustering,
    RecurrenceManifold,
    ShrecPipeline,
)
from shrec.recurrence.connectivity import (
    CommonNeighborsConnectivity,
    ExpKernelConnectivity,
    PrecomputedConnectivity,
    SimplicialConnectivity,
)
from shrec.reconstruct import (
    FiedlerReconstructor,
    IsomapReconstructor,
    LeidenReconstructor,
    UnionFindReconstructor,
)
from shrec.utils import common_neighbors_ratio


def _cos(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return abs(float(np.dot(a, b)))


# --- MM37 — preset ≡ ShrecPipeline -----------------------------------------

class TestPresetPipelineEquivalence:
    """MM37 (must) — each named model equals the explicit `ShrecPipeline`
    composition of its stages, with matched parameters. This is the contract
    that the presets are faithful thin wrappers over the spine and did not
    silently drift from it (0008)."""

    def test_clustering_equals_pipeline(self, square_wave_responses):
        X, _ = square_wave_responses(T=300, n_responses=8)
        preset = RecurrenceClustering(random_state=1).fit(X).labels_
        pipe = ShrecPipeline(
            connectivity=SimplicialConnectivity(k=10, tol=1e-5, time_exclude=0),
            reconstructor=LeidenReconstructor(
                resolution=1.0, objective="modularity",
                method="graspologic", random_state=1,
            ),
        ).fit(X).labels_
        assert adjusted_rand_score(preset, pipe) == pytest.approx(1.0, abs=1e-9)

    def test_manifold_equals_pipeline(self, square_wave_responses):
        X, _ = square_wave_responses(T=250, n_responses=8)
        preset = RecurrenceManifold(random_state=1).fit(X).labels_
        pipe = ShrecPipeline(
            connectivity=SimplicialConnectivity(k=10, tol=1e-5, time_exclude=0),
            reconstructor=FiedlerReconstructor(n_components=1),
        ).fit(X).labels_
        assert _cos(preset, pipe) > 0.999

    def test_classical_equals_pipeline(self, square_wave_responses):
        X, _ = square_wave_responses(T=250, n_responses=8)
        preset = ClassicalRecurrenceClustering().fit(X).labels_
        pipe = ShrecPipeline(
            connectivity=ExpKernelConnectivity(
                scale=1.0, order=500.0, time_exclude=0,
                sparsify=True, tolerance=0.01, weighted=True,
            ),
            reconstructor=UnionFindReconstructor(),
        ).fit(X).labels_
        np.testing.assert_array_equal(preset, pipe)

    def test_hirata_nomura_equals_pipeline(self, square_wave_responses):
        X, _ = square_wave_responses(T=200, n_responses=6)
        preset = HirataNomuraIsomap(n_components=2).fit(X).labels_
        pipe = ShrecPipeline(
            connectivity=CommonNeighborsConnectivity(percentile=0.1),
            reconstructor=IsomapReconstructor(n_components=2),
        ).fit(X).labels_
        np.testing.assert_allclose(preset, pipe, atol=1e-9)


# --- MM40 — stage strategy contracts ----------------------------------------

def _two_block_affinity(p1=8, p2=12, bridge=1e-2):
    """Connected weighted two-block affinity with unit diagonal (the shape a
    simplicial consensus has)."""
    n = p1 + p2
    A = np.zeros((n, n))
    A[:p1, :p1] = 1.0
    A[p1:, p1:] = 1.0
    A[p1 - 1, p1] = A[p1, p1 - 1] = bridge
    return A


class TestReconstructorContract:
    """MM40 — mechanical pins on the stage strategies: output shapes, the
    discrete reconstructors' derived attributes, and the injection seam."""

    def test_fiedler_shapes(self):
        A = _two_block_affinity()
        n = A.shape[0]
        assert FiedlerReconstructor(n_components=1)(A).shape == (n,)
        assert FiedlerReconstructor(n_components=2)(A).shape == (n, 2)

    def test_leiden_shape_and_extra_attrs(self):
        A = _two_block_affinity()
        rec = LeidenReconstructor(random_state=1)  # graspologic requires a positive seed
        labels = rec(A)
        assert labels.shape == (A.shape[0],)
        extra = rec.extra_attrs(labels)
        assert set(extra) == {"n_clusters", "has_unclassified"}
        assert extra["n_clusters"] >= 1

    def test_unionfind_shape_and_no_extra_attrs(self):
        A = _two_block_affinity()
        rec = UnionFindReconstructor()
        labels = rec(A > 0)  # union-find consumes a binary adjacency
        assert labels.shape == (A.shape[0],)
        assert rec.extra_attrs(labels) == {}

    def test_isomap_shape(self):
        # Isomap consumes a dissimilarity; feed the HN common-neighbour matrix.
        sim = common_neighbors_ratio((_two_block_affinity() > 0).astype(int))
        out = IsomapReconstructor(n_components=2)(sim)
        assert out.shape == (sim.shape[0], 2)

    def test_precomputed_connectivity_returns_fixed_matrix(self):
        A = _two_block_affinity()
        conn = PrecomputedConnectivity(A)
        # Ignores its input and returns the injected matrix.
        np.testing.assert_array_equal(conn(np.zeros((A.shape[0], 1))), A)
        np.testing.assert_array_equal(conn(np.zeros((3, 5))), A)


# --- MM41 — off-preset combinations recover a known driver (slow) -----------

def _smooth_driver_ensemble(T=400, n_responses=12, coupling=0.4, noise=0.03, seed=0):
    """Chaotic logistic responses forced by a smooth sinusoidal driver, lightly
    observed through noise. Returns (X, driver)."""
    rng = np.random.default_rng(seed)
    driver = 0.5 + 0.4 * np.sin(2 * np.pi * np.arange(T) / 90.0)
    r_values = rng.uniform(3.7, 3.9, size=n_responses)
    X = np.empty((T, n_responses))
    for k, r in enumerate(r_values):
        x = np.empty(T)
        x[0] = rng.uniform(0.1, 0.9)
        for t in range(T - 1):
            x[t + 1] = np.clip(r * x[t] * (1 - x[t]) + coupling * driver[t], 0.0, 1.0)
        X[:, k] = x + noise * rng.standard_normal(T)
    return X, driver


@pytest.mark.slow
class TestOffPresetComboRecovery:
    """MM41 (slow, characterisation) — the *point* of the refactor is that
    unnamed combinations work. Swap the connectivity under a spectral /
    community reconstructor and confirm the driver is still recovered. These
    are capability bands, not tight oracles."""

    def test_expkernel_plus_fiedler_recovers_continuous_driver(self):
        X, driver = _smooth_driver_ensemble()
        v = ShrecPipeline(
            connectivity=ExpKernelConnectivity(sparsify=False),
            reconstructor=FiedlerReconstructor(n_components=1),
        ).fit(X).labels_
        assert abs(spearmanr(v, driver).correlation) > 0.6

    def test_expkernel_plus_leiden_recovers_period_two(self, square_wave_responses):
        X, z = square_wave_responses(T=600, n_responses=16, coupling=0.5)
        labels = ShrecPipeline(
            connectivity=ExpKernelConnectivity(sparsify=False),
            reconstructor=LeidenReconstructor(random_state=1),
        ).fit(X).labels_
        assert adjusted_rand_score(labels, (z > 0.5).astype(int)) > 0.6
