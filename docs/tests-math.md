# shrec — math-correctness test catalog

The `MM<n>` ids are referenced by the docstring of every math-test in
`tests/`. Tests in §5b.1–§5b.5 below probe SHREC against analytical
results, limiting cases, and explicit paper claims — i.e. they catch
*algorithmic* drift that mere structural / shape tests cannot.

"must-have" = catches a class of bugs that the structural tests cannot.
"regression" = guards a property but won't fail on a fresh implementation
if math is right.

Status legend:

- ✅ **green** — implemented, currently passing.
- ⚠️ **xfail** — implemented as a documented `@pytest.mark.xfail`;
  failure is a known property of the current code, not a regression.
- ⏳ **deferred** — not yet implemented; tracked here for future work.

---

## §5b.1 — Closed-form unit tests of the inner math

These don't need a time series at all; they probe individual operators
with inputs whose answers can be written down.

| Test | Status | Module / Oracle |
|------|--------|------------------|
| **MM1 (must)** | ✅ | `recurrence/simplicial.py:fit_rho_sigma` — defining equation `\|Σ_m exp[-ReLU(d_im − ρ)/σ] − log₂ k\| < 1e-6` on every row. **Caught a real bug**: `fsolve` reported `ier=1` while stalling at the initial guess on ~6% of rows (residual ≈ 2.8). Fixed by a bracketed `brentq` (the equation is monotone in σ). See `tests/test_recurrence_simplicial.py` and `docs/math-learning-notes.md`. |
| **MM2 (must)** | ✅ | Same — **scale invariance**: `fit_rho_sigma(α · d_row, k) = (α·ρ, α·σ)` so affinity is unchanged. Verified on both `(ρ, σ)` and the full `dataset_to_simplex` output. |
| **MM3 (must)** | ✅ | Same — **diagonal**: `A_ii = 1` exactly (`d_ii=0 ⇒ exp(0)=1`, fuzzy union `1+1−1=1`). |
| **MM4 (must)** | ✅ | `dataset_to_simplex` symmetrisation: `A + Aᵀ − A∘Aᵀ ∈ [0,1]^(N×N)` and symmetric. |
| **MM5 (must)** | ✅ | `fit_rho_sigma` vs `umap.umap_.fuzzy_simplicial_set`, reconciled. The historical divergence was (1) a real **off-by-one bug** in `dataset_to_simplex` — it inf-filled the self-distance *and then* sliced `[1:k+1]`, double-skipping so ρ became the *second*-nearest neighbour; Appendix B (`ρᵢ ≡ minₘ{dᵢₘ}`), umap, and the original author's own `# nearest neighbor` / `# drop self` comments all say *nearest*. Fixed to `[:k]`; ρ now matches umap exactly. (2) A documented convention: umap's `n_neighbors=k` counts the query point, so it sums k-1 real neighbours toward log₂(k); adopting that, σ matches umap to ~1e-6. See `tests/test_recurrence_simplicial.py` and Round 11 of `docs/math-learning-notes.md`. |
| **MM6 (must)** | ✅ | `recurrence/kernel.py:data_to_connectivity` p-norm limits: the ensemble aggregation is a power-mean `(mean_i a_i**ord)**(1/ord)` of per-channel kernels — `ord=1` → arithmetic mean; `ord→∞` → elementwise max = min-over-channels (Sauer `inf_k`). L∞ limit verified on well-conditioned entries (small affinities underflow at high ord). See `tests/test_recurrence_kernel.py`. |
| MM7 | ✅ | Recurrence **rigid-motion invariance** — lifted from raw `cdist` to the affinity level: `dataset_to_simplex(X·Q + b) = dataset_to_simplex(X)` for orthogonal `Q` (rotations + reflections) and translation `b`, to `atol=1e-6`. Exercises the real ρ/σ pipeline (complements the scale-invariance oracle MM2); a companion check pins the underlying `cdist` isometry to `atol=1e-10`. See `tests/test_recurrence_distance.py`. |
| MM8 | ✅ | Metric contract of the distance primitive: triangle inequality `d(i,k) ≤ d(i,j) + d(j,k)` over all triples, plus zero/symmetry/non-negativity. The ρ-as-nearest-neighbour / σ-bandwidth logic assumes a true metric. See `tests/test_recurrence_distance.py`. |
| MM9 | ✅ | `utils/metrics.py:sparsify` (the catalog's `sparsify_by_quantile`): achieved zero-fraction ≥ requested `sparsity` (the `interpolation="higher"` quantile rounds the threshold up so ties never undershoot), idempotent at the same target (binary output is a fixed point), binary-valued when unweighted, monotone in the target. See `tests/test_metrics.py`. |
| **MM35** | ✅ | `recurrence/kernel.py:distance_to_connectivity` bracket robustness — the fixed bracket `[1e-16, dscale]` assumed a sign change that fails when the requested sparsity is below the `1/N` diagonal floor (raised `ValueError` for small N). Now guards the infeasible case (warn + sharpest kernel) and expands the upper bracket; same fix discipline as `fit_rho_sigma`. See `tests/test_recurrence_kernel.py`. |
| **MM10 (must)** | ✅ | `graph/communities.py:_leiden` — backend agreement on the barbell graph (ARI=1 across backends). graspologic leg always runs; igraph/leidenalg/cdlib gated by `importorskip`. Also locks the igraph `resolution`-forwarding fix (the branch had hardcoded `resolution_parameter=1.0`). See `tests/test_graph_communities.py`. |
| MM11 | ✅ | `graph/unionfind.py` parity with `scipy.cluster.hierarchy.DisjointSet`: identical connected-components partition over random merge sequences, and `solve_union_find` returns each group's full transitive closure (cross-checked against scipy `.subset`). Clears the swap-in-scipy-behind-an-adapter path the module's docstring flags. See `tests/test_graph_unionfind.py`. |
| MM12 | ✅ | `utils/graph_tools.common_neighbors_ratio` — vectorised matches loop on random binary matrices to `atol=0`; K_n with self-loops returns zero. See `tests/test_graph_adjacency.py`. |
| MM38 | ✅ | `recurrence/consensus.py:_aggregate` aggregation limits (the simplicial-consensus analogue of MM6): `pnorm:1 ≡ mean`, `mean` is the arithmetic mean, `max` is the elementwise max of the per-response stack, and `pnorm:p → max` as `p→∞` on well-conditioned entries (tiny-max entries converge slower — same underflow caveat as MM6). Also pins the wiring: `data_to_connectivity2(aggregation="pnorm:1")` reproduces the default mean on a real ensemble. See `tests/test_recurrence_consensus.py`. |
| MM39 | ✅ | `metric` plumbing (0008): the distance `metric` reaches the per-response `cdist`, so a non-euclidean metric (`cityblock`) measurably changes the affinity (not silently ignored) while `metric="euclidean"` reproduces the default — and the MM3/MM4 invariants (unit diagonal, symmetry, `[0,1]`) survive any `cdist` metric. See `tests/test_recurrence_consensus.py`. |

---

## §5b.2 — Algorithmic invariances and equivariances

Properties the *whole* `fit_predict` pipeline must satisfy on
principled grounds (recurrence is undirected; no preferred response
order; etc.).

| Test | Status | Property |
|------|--------|----------|
| **MM13 (must)** | ✅ | Response-permutation invariance: `model.fit(X[:, perm])` gives the same labels (ARI=1) as `model.fit(X)`. See `tests/test_models_invariances.py`. |
| **MM14 (must)** | ✅ approximate | Time-reversal symmetry as **approximate** (ARI > 0.65 for clustering, `|cos| > 0.7` for manifold). The doc's exact-equality claim does not hold once delay embedding enters: forward embedded points carry past lags, reversed points carry "future" lags, so the per-point coordinates aren't byte-equal. The driven logistic map is non-reversible, so reverse-vs-truth ARI ≈ 0.77 vs forward ≈ 0.98; the clustering threshold was lowered 0.85→0.65 after the MM5 ρ fix (the sharper, paper-faithful kernel surfaces this real asymmetry rather than masking it). Still guards against a gross directional leak. |
| MM15 | ✅ | Full-pipeline invariance to a **per-channel affine** map `x_k → a_k x_k + b_k` (`a_k>0`): `standardize=True` (default) z-scores each response, so MM2 (affinity scale-invariance) + standardisation = full `fit_predict` invariance. ARI=1 (clustering), `|cos|>0.999` (manifold). See `tests/test_models_invariances.py`. |
| MM16 | ✅ | **Duplication idempotence**: `fit([X, X]) = fit(X)`. The two *label-level* tests pin driver-invariance under duplication — which holds for any uniform consensus (mean **or** raw sum), since downstream Leiden/Fiedler are scale-invariant, so they do *not* pin the `1/K` (cf. the Appendix-B printed-sum typo, decision 0002). A third, *matrix-level* test (`test_consensus_mean_is_idempotent`) pins the `1/K` specifically: the consensus affinity is idempotent under ensemble duplication for a mean but would double for a sum. Stated on the ensemble, not a single channel, because `standardize_ts` squeezes an `(T,1)` input to 1-D. See `tests/test_models_invariances.py`. |
| MM17 | ✅ | **Constant-response rejection**: `recurrence/kernel.py:data_to_connectivity` detects channels equal to their first timepoint (degenerate `surprise=0/0`), warns, and drops them — output equals the surviving non-constant channels exactly; no warning when all channels vary. See `tests/test_recurrence_kernel.py`. |
| MM18 | ✅ (as MM31) | Determinism w.r.t. `random_state` — and crucially, *no* global RNG mutation. See `tests/test_models_base.py::TestRngIsolation`. |

---

## §5b.3 — Limiting-case oracles (closed-form ground truth)

Inputs constructed so the answer is provably the one we want.

| Test | Status | Input / Oracle |
|------|--------|------------------|
| **MM19 (must)** | ✅ near-exact | Sauer limit, period-2 driver: N=20 logistic responses, zero noise, T=1000 → `RecurrenceClustering().fit(X)` gives ARI > 0.95 (≈ 0.99). Exact ARI=1 is the asymptotic claim; the residual ~1% is Leiden boundary over-segmentation. See `tests/test_models_recurrence_clustering.py`. |
| **MM20 (must)** | ⚠️ xfail | Sauer limit, period-4 driver → ARI ≈ 0.50. **Diagnosed as representational, not a Leiden artifact** (companion characterisation test `test_period_four_is_not_separable_in_graph`): a modularity-resolution sweep jumps 2 communities (res≤1) → ~1000 singletons (res≥2) with no stable 4-community regime; oracle spectral k-means=4 on the consensus Laplacian also gives ARI≈0.5 across N∈[20,100], coupling∈[0.5,1]; continuous RecurrenceManifold gives Spearman |ρ|≈0.38. The four levels collapse into a low/high 2-way split — closing MM20 needs a richer recurrence representation, not a clustering tweak. |
| MM21 | ✅ characterisation (slow) | Period-8 stochastically-forced driver. The catalog's ARI > 0.85 target is **not met** (measured ARI ≈ 0.64, continuous |ρ| ≈ 0.44): the same representational limit as MM20, *worse* with more driver levels. Pinned as a partial-collapse band (`0.25 < ARI < 0.85`, `|ρ| < 0.85`) — clearly below target (limit is real) yet above chance (structure partially present). Fires if a future representation lifts recovery past 0.85. See `tests/test_models_recurrence_clustering.py`. |
| **MM22 (must)** | ✅ | Block-stochastic affinity: hand-construct `A = block_diag(J_p1, J_p2)` (unequal sizes) with a small bridge, assert RecurrenceManifold output `|cos|` > 0.99 against the analytical Fiedler vector. Distinguishes Fiedler from second SVD vector on irregular graphs. See `tests/test_models_recurrence_manifold.py`. |
| **MM33 (must)** | ✅ | `RecurrenceManifold` connectivity guard: a (nearly) disconnected consensus graph has `λ₂ ≈ 0`, so the Fiedler eigenvector degenerates into a component indicator. `fit` must warn (and the well-connected MM22 case must not). Scale-free threshold `λ₂ ≤ 1e-10·Σdegree`. See `tests/test_models_recurrence_manifold.py`. |
| MM34 | ✅ | `RecurrenceManifold(normalize_laplacian=True)` — opt-in NCut / random-walk normalisation (generalised `L v = λ D v`), the paper's "preconditioning" remedy for response bias. Must still recover a clean block split and must measurably differ (`|cos| < 0.95`) from the unnormalised default under degree heterogeneity. See `tests/test_models_recurrence_manifold.py`. |
| MM23 | ✅ | Cycle-graph `C_n` affinity: the Fiedler eigenvalue is **doubly degenerate**, so the Fiedler vector is fixed only up to sign *and* phase — it's any unit vector in the first-harmonic plane span{`cos(2π i/n)`, `sin(2π i/n)`}. Oracle is the precise subspace claim: recovered eigenvector(s) lie in that plane (>0.999 of norm) at the first harmonic, not a higher one (<0.05 in the 2nd). See `tests/test_models_recurrence_manifold.py`. |
| MM24 | ✅ | Identity measurement `x_k = z` (smooth aperiodic driver, light noise) — Spearman `|ρ| > 0.95` (measured ≈ 0.997). Uses N>1 noisy copies, not the literal N=1 (which `standardize_ts` squeezes to 1-D, cf. MM16). See `tests/test_driver_recovery.py`. |
| MM25 | ✅ | Linear measurement `x_k = a_k z + b_k` — `|ρ| > 0.9` (standardisation removes the per-sensor gain/offset). See `tests/test_driver_recovery.py`. |
| MM26 | ✅ | Monotone nonlinear measurement `x_k = tanh(z/σ_k)` — `|ρ| > 0.8`; recovery up to a monotone transform, which Spearman is invariant to. See `tests/test_driver_recovery.py`. |
| MM41 | ✅ (slow) | **Off-preset combo recovery** (plan 0009): the stage-strategy refactor (0008) is only useful if *unnamed* combinations work. `ExpKernelConnectivity(sparsify=False)` + `FiedlerReconstructor` recovers a smooth continuous driver (Spearman `|ρ| > 0.6`); `ExpKernelConnectivity` + `LeidenReconstructor` recovers the period-2 driver (ARI > 0.6). Characterisation bands, not tight oracles. Records the *sensible-combo* contract: `Fiedler`/`Leiden` take an affinity (any connectivity); `UnionFind` needs a sparse/binary graph (`ExpKernelConnectivity(sparsify=True)`); `Isomap` wants a dissimilarity. See `tests/test_pipeline.py`. |
| MM42 | ✅ | **`SimplicialConnectivity(sparsify=…)`** (0009 follow-up): the optional sparsify step thresholds the dense fuzzy affinity to the target sparsity `1 − tolerance` (achieving *at least* that zero-fraction, cf. MM9), so the canonical adaptive graph can feed a reconstructor that needs a sparse/binary input (the seam that made `Simplicial + UnionFind` a non-degenerate combo). `sparsify=False` (default) leaves the output byte-identical; `weighted` toggles magnitude-preserving vs 0/1; symmetry survives. See `tests/test_pipeline.py`. |

---

## §5b.4 — Paper-claim regression tests (quantitative scaling laws)

Slower; intended for `-m slow` / nightly CI.

| Test | Status | Claim |
|------|--------|-------|
| **MM27 (must)** | ✅ (slow) | β-accuracy scaling (Appendix E.2): `Acc(NT/τ) = Acc_max (1 − exp(−β √(NT/τ)))`. Fit to a Spearman-accuracy-vs-N sweep on a lightly-noised (σ=0.05) continuous-driver logistic ensemble via `RecurrenceManifold` (the continuous path sidesteps the MM20 discrete-ARI collapse and is smooth to fit). Asserts β>0, Acc_max∈[0.6,1], clear small-N→large-N gain, and that the form beats a flat baseline (R²>0.5). See `tests/test_scaling_laws.py`. |
| MM28 | ✅ (slow) | Percolation order parameter (Appendix E.3): `T_LCC/T` (largest-connected-component fraction of the consensus graph A) is monotone non-increasing in N — "percolation loss precedes accurate reconstruction" (Fig 6). Reproduced on the **simplicial consensus** (the paper's A, not the Sauer graph) at an **absolute** edge threshold (a *quantile* threshold re-percolates as the weight distribution shifts with N) in the **weak-coupling** regime (κ=0.1, so single responses are individually ambiguous and consensus does the resolving — strong coupling is N-invariant). Seed-averaged (10) curve: 0.85 → 0.27, Spearman(N, LCC) = −1.0; asserts Spearman < −0.8 and drop > 0.3. See `tests/test_scaling_laws.py` and Round 12 of `docs/math-learning-notes.md`. |
| MM29 | ✅ | HN-Isomap baseline: the consensus similarity `HirataNomuraIsomap` feeds Isomap (`metric='precomputed'`) is a valid dissimilarity — symmetric, zero-diagonal, non-negative, ≤1 — and `fit` yields a finite `(T, n_components)` embedding. (Element-wise `common_neighbors_ratio` correctness is MM12.) See `tests/test_models_hirata_nomura.py`. |

---

## §5b.5 — Sklearn-contract tests

Cheap mechanical pinning.

| Test | Status | Property |
|------|--------|----------|
| MM30 | ✅ | `set_params(**get_params())` is the identity on the model state. See `tests/test_models_base.py::TestSklearnContract`. |
| MM31 | ✅ | Constructing a model does not change `np.random.get_state()`. See `tests/test_models_base.py::TestRngIsolation`. |
| MM32 | ✅ (merged with MM30) | `set_params`/`get_params` round-trip across the four models. |
| MM36 | ✅ | `RecurrenceManifold` eigenvector shape contract: `subset_by_index=[1, n_components]` returns exactly `n_components` non-trivial eigenvectors, so `labels_` is `(T,)` for `n_components=1` and `(T, n_components)` otherwise. Pins against an index-range off-by-one. See `tests/test_models_recurrence_manifold.py`. |
| **MM37 (must)** | ✅ | **Preset ≡ `ShrecPipeline`** (plan 0009): each of the four named models equals the explicit `ShrecPipeline(connectivity=…, reconstructor=…)` composition of its stages with matched parameters (clustering ARI=1, manifold `\|cos\|>0.999`, classical labels identical, HN `allclose`). Pins that the presets are faithful thin wrappers over the spine and did not silently drift from it (0008). See `tests/test_pipeline.py`. |
| MM40 | ✅ | **Stage-strategy contracts** (plan 0009): `Reconstructor` output shapes (`Fiedler`/`Isomap` `(T,)` vs `(T,n_components)`; `Leiden`/`UnionFind` `(T,)`), `LeidenReconstructor.extra_attrs` sets `n_clusters`/`has_unclassified` (others `{}`), and `PrecomputedConnectivity(A)` returns `A` ignoring its input (the injection seam). See `tests/test_pipeline.py`. |

---

## Coverage summary

| Section | Total | Green | xfail | Deferred |
|---------|-------|-------|-------|----------|
| §5b.1 inner math      | 15 | 15 | 0 | 0  |
| §5b.2 invariances     | 6  | 6 | 0 | 0  |
| §5b.3 limiting cases  | 11 | 10 | 1 | 0  |
| §5b.4 scaling laws    | 3  | 3 | 0 | 0  |
| §5b.5 sklearn contract| 6  | 6 | 0 | 0  |
| **total**             | 41 | 40 | 1 | 0  |

MM1, MM3 and MM4 also have **Hypothesis property-based** generalisations
(`TestSimplexInvariantsPropertyBased`) that assert the invariants over
machine-generated clouds and independently rediscover the tied-neighbourhood
σ-degeneracy. Hypothesis additionally surfaced a *near*-tied regime (neighbours
separated by ε ≪ scale → root σ ~ ε, with `f'(σ) ~ 1/σ`): `brentq` locates the
root to `xtol` but the function residual there is `~f'·xtol ~ 1/ε`, so a correct
solve can exceed a naïve `1e-6` residual bound. The property test now accepts
that scale-invariant steep-root branch (`σ ≪ neighbourhood scale`) — distinct
from the fsolve stall (`σ = ρ`, residual ≈ 2.8) it still catches. See Round 7 of
`docs/math-learning-notes.md`.

The "must" tests (MM1–MM6, MM10, MM13–MM14, MM19–MM22, MM27, MM33) are
the minimum to call the algorithm green. Of the 14 must-tests, **13 are
green and 1 is xfail-documented — none are deferred**. The inner-math
closed-form checks MM1–MM4 are green (MM1 surfaced and fixed the `fsolve`
σ-solve stall); MM10 (backend agreement) and MM33 (connectivity guard)
landed alongside the igraph-`resolution` and disconnected-graph fixes;
MM6 (kernel p-norm limits) and MM27 (accuracy scaling law) close the set;
MM5 (umap parity) surfaced and fixed a real ρ off-by-one and now matches umap.
The only open must-item is the documented xfail MM20 (period-4 representational
collapse).

**The catalog is now fully closed — 0 deferred items.** All sections green
except the single documented MM20 xfail. The post-PR follow-up batches: batch 1 =
MM7/MM8/MM9/MM11; batch 2 = MM15/MM16/MM17; batch 3 = MM23 (cycle-graph Fiedler)
+ MM24/MM25/MM26 (driver recovery under identity/linear/monotone measurement) +
MM21 (period-8, landed as a representational-limit *characterisation* — the naive
ARI>0.85 target is provably unreachable, same limit as MM20); batch 4 = MM29
(HN-Isomap baseline sanity); the MM5 ρ off-by-one fix (see §5b.1); and finally
MM28 (percolation order parameter, Appendix E.3) — reproduced on the simplicial
consensus at an *absolute* threshold in the weak-coupling regime (the earlier
calibration block was a quantile threshold re-percolating as the weight
distribution shifts with N). The one open xfail (MM20 period-4 representational
collapse) is documented, not a regression.
