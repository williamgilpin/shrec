# 0009 — Coverage plan for the composable pipeline (tests / benchmarks / examples)

**Status:** executed (2026-06-17) · **Date:** 2026-06-16
**Follows:** 0008 (the refactor that enabled these combinations).

> ## Outcome (2026-06-17)
>
> All three tiers landed. Suite: **177 passed, 6 skipped, 1 xfailed** (172
> `-m "not slow"` + 5 slow).
>
> - **Tests** — `tests/test_pipeline.py` (MM37 preset≡pipeline for all four
>   models; MM40 stage contracts; MM41 slow off-preset recovery) and
>   `tests/test_recurrence_consensus.py` (MM38 aggregation limits; MM39 metric
>   plumbing). `tests/test_plotting.py` covers the viz helpers. Catalog updated
>   to 41 oracles (40 green / 1 xfail).
> - **Benchmark** — `benchmarks/pipeline_combo_sweep.py` → `pipeline_combo_scores.csv`.
> - **Examples** — `examples/quickstart.py`, `examples/pipeline_composition.py`
>   (runnable scripts, notebook-ready), `examples/README.md`; plus the
>   `shrec.plotting` module (E3) — `plot_driver_overlay`, `plot_recurrence_matrix`.
>
> ### What execution surfaced (not in the original plan)
>
> 1. **The "sensible-combo contract" is real and load-bearing.** Reconstructors
>    have implicit input semantics: `Fiedler`/`Leiden` take an affinity (any
>    connectivity); `UnionFind` needs a *sparse/binary* graph — a dense
>    affinity makes every node mutually reachable, so `Simplicial+UnionFind`
>    collapses to one component (ARI≈0, confirmed in the benchmark); `Isomap`
>    wants a dissimilarity. MM41 / the benchmark / the example all now state
>    this. The plan's casually-listed `Simplicial+UnionFind` combo is a
>    *non-sequitur* without sparsification.
> 2. **The canonical presets are not universally best.** On the clean discrete
>    period-2 task, `ExpKernel+Leiden` (ARI≈0.97) beats `Simplicial+Leiden`
>    (≈0.53–0.70) — the Sauer-limit regime favours the sharp exp kernel;
>    `Simplicial+Fiedler` wins the continuous task (|ρ|≈0.82). Good
>    default-justification data.
>
> ### Discovered follow-up
>
> - **Add an optional `sparsify` to `SimplicialConnectivity`** so
>   `Simplicial+UnionFind` becomes meaningful and the combo grid has no
>   degenerate cell. Small, genuinely useful; was deferred as a feature, not
>   coverage. **Done (2026-07-05):** `SimplicialConnectivity(sparsify=True,
>   tolerance=…, weighted=…)` mirrors `ExpKernelConnectivity`; thresholds the
>   dense affinity to `1 − tolerance`; default path byte-identical; pinned by
>   MM42 (`tests/test_pipeline.py`). Note the recovered *labels* on
>   `Simplicial+UnionFind` are still muddied by the separate classical
>   union-find label quirk (below) — sparsify fixes the graph, not the
>   labeler. Real DTW and that union-find quirk remain the other two known
>   out-of-scope items.

## Why

The 0008 refactor *enabled* arbitrary `(Connectivity × Reconstructor)`
composition, new consensus aggregation modes (`mean`/`max`/`pnorm:<p>`), and a
pluggable distance `metric` — but nothing exercises them. The four preset
paths stay green; the **new** surface (`ShrecPipeline`, the off-preset combos,
the aggregation/metric knobs) is currently untested, unbenchmarked, and
undemonstrated. This plan scopes the coverage that locks the flexibility in
and shows people how to use it.

Three tiers, in priority order: **(1) tests** (cheap, exact, must-have before
the combos are "real"), **(2) benchmarks** (which combos are actually good on
the paper's tasks), **(3) examples** (make it discoverable).

---

## 1. Tests

Each claims a fresh `MM<n>` id and a row in `docs/tests-math.md`. New ids
continue from MM36.

### MM37 (must) — preset ≡ pipeline equivalence
The whole refactor rests on "the four named models are faithful thin presets."
Pin it: for matched params, `ShrecPipeline(connectivity=…, reconstructor=…)`
must equal the preset.
- `ShrecPipeline(SimplicialConnectivity(k=10, tol=1e-5),
  LeidenReconstructor(random_state=1))` vs `RecurrenceClustering(random_state=1)`
  → ARI = 1 on the `square_wave_responses` fixture.
- Same for `SimplicialConnectivity` + `FiedlerReconstructor` vs
  `RecurrenceManifold` → `|cos| > 0.999`.
- `ExpKernelConnectivity(sparsify=True)` + `UnionFindReconstructor` vs
  `ClassicalRecurrenceClustering` → identical labels.
File: `tests/test_pipeline.py` (new). Cheap, exact, *must-have* — this is the
contract that the presets didn't silently drift from the spine.

### MM38 — simplicial-consensus aggregation limits
The analogue of MM6 (which covers the *exp-kernel* p-norm) for the new
`SimplicialConnectivity(aggregation=…)`:
- `pnorm:1` ≡ `mean` (elementwise, atol≈1e-12);
- as `p→∞`, `pnorm:<p>` → `max` (elementwise) on well-conditioned entries
  (small affinities underflow at high p — same caveat as MM6);
- `max` aggregation is the elementwise max of the per-response stack.
File: `tests/test_recurrence_consensus.py` (new — there is no dedicated
consensus test module yet; today the mean path is only covered indirectly).
Cheap, exact.

### MM39 — distance-metric plumbing
`metric` now reaches `cdist` (0008). Pin it both ends:
- `SimplicialConnectivity(metric="euclidean")` output == the default;
- a non-euclidean metric (`"cityblock"`, `"cosine"`) produces a *different*
  `A` (so the knob isn't silently ignored), still symmetric with unit diagonal
  (MM3/MM4 invariants hold under any metric).
File: `tests/test_recurrence_consensus.py`. Cheap.

### MM40 — reconstructor contract
Mechanical pins on the `Reconstructor` strategies:
- each returns labels of shape `(T,)` (or `(T, n_components)` for Fiedler/Isomap
  `n_components>1`);
- `LeidenReconstructor.extra_attrs` sets `n_clusters` / `has_unclassified`;
  the others return `{}`;
- `PrecomputedConnectivity(A)(X) is A` for any `X` of the right leading dim.
File: `tests/test_pipeline.py`. Cheap mechanical (sklearn-contract tier).

### MM41 (characterisation, slow) — off-preset combos recover a known driver
The point of the flexibility is that *unnamed* combinations work. Light
capability checks on simple systems, not accuracy claims:
- `SimplicialConnectivity` + `UnionFindReconstructor` (adaptive graph, Sauer
  reconstruction) on the noise-free **period-2** limit → ARI > 0.9;
- `ExpKernelConnectivity(sparsify=False)` + `FiedlerReconstructor` (Sauer
  kernel, continuous driver) on a smooth driver → Spearman `|ρ| > 0.8`.
File: `tests/test_pipeline.py`, `@pytest.mark.slow`. Guards that the cross
combos don't degenerate; flags if a future change breaks one.

> Note: MM37/40 are the *must-have* gate (they pin the refactor's contract);
> MM38/39/41 raise confidence in the new knobs.

---

## 2. Benchmarks

Goal: answer "which combinations are actually worth using?" and confirm the
canonical presets are the right defaults on the paper's tasks. Reuse the
existing harness (`benchmarks/dynamical_systems.py:DrivenLogistic` /
`DrivenLorenz`, `benchmarks/benchmark_utils.py`).

### B1 — combo grid (the headline benchmark)
A small `(connectivity × reconstructor)` grid scored against the true driver
on `DrivenLogistic` (discrete, period-2/4) and `DrivenLorenz` (continuous,
Rössler):
| connectivity \ reconstructor | Leiden | Fiedler | UnionFind | Isomap |
|---|---|---|---|---|
| Simplicial | *canonical* | *canonical* | new | new |
| ExpKernel  | new | new | *classical* | new |
Report ARI (discrete) / Spearman (continuous) vs N. **Expected finding:** the
canonical Simplicial+Leiden / Simplicial+Fiedler win; document where (and
whether) any off-preset combo is competitive. Output a CSV alongside the
existing `acc_scores.csv`. New file: `benchmarks/pipeline_combo_sweep.py`.

### B2 — aggregation sweep (ties to MM28)
On the weak-coupling percolation regime (the MM28 setting), sweep
`aggregation ∈ {mean, pnorm:2, pnorm:10, max}` vs N. Question: does sharper,
Sauer-like aggregation (large `p`) help or hurt consensus recovery vs the
paper's mean? Reports accuracy and the `T_LCC/T` order parameter per mode.

### B3 — metric sweep
On a system where a non-euclidean metric is *principled* (e.g. phase/angular
responses → `cosine`/`cityblock`), compare `metric` choices. Demonstrates the
metric seam has a real use, not just an API knob.

All three are scripts (not notebooks) that write a CSV; keep them out of the
pytest path (they're sweeps). They double as regression baselines.

---

## 3. Examples

There is no `examples/` dir yet (the pre-release "usability strand",
`docs/pre-release-notes.md`, scopes the first two). These make the new
capability discoverable.

### E1 — `examples/quickstart.ipynb`
Already on the release punch list. `X = load_lorenz()` →
`RecurrenceManifold().fit_predict(X)` → driver-overlay plot in ≤ 20 cells. The
baseline "does it work at all" artifact; now also show the one-line
`from shrec import RecurrenceManifold`.

### E2 — `examples/pipeline_composition.ipynb` (the 0008 showcase)
The headline example for the new architecture:
1. the canonical preset, then the *same* result via explicit
   `ShrecPipeline(SimplicialConnectivity(), FiedlerReconstructor())`;
2. swap one stage at a time — `UnionFindReconstructor`, then
   `ExpKernelConnectivity` — plotting the recovered driver after each swap, so
   the one-line swap is visible;
3. `aggregation="max"` vs `"mean"` and a non-euclidean `metric`, side by side;
4. `PrecomputedConnectivity(A)` to drive reconstruction from a user-supplied
   graph.
This is the single artifact that turns "the code is flexible now" into
"here's how you use the flexibility."

### E3 — depends on `shrec.plotting`
E1/E2 want `plot_driver_overlay(z_true, z_pred)` and
`plot_recurrence_matrix(A)` (also on the usability strand). Build those two
helpers first; the examples then stay short. This is the natural place to land
that module.

---

## Sequencing / dependencies

1. **MM37 + MM40** (preset≡pipeline + reconstructor contract) — do first; they
   gate everything as the correctness floor for the refactor.
2. **MM38 + MM39** (aggregation + metric) — close behind; they pin the new
   knobs.
3. **B1** (combo grid) — once tests pass, it tells us which combos to feature.
4. **`shrec.plotting` (E3) → E1 → E2** — examples last, informed by B1's
   findings about which combos are worth showing.
5. **MM41** (slow combo capability) — fold into the slow CI lane.

## Out of scope here

Real DTW (`metric="dtw"` still unimplemented — only `cdist` metrics work) and
the classical union-find label quirk (decision 0008 outcome) are separate
items, not part of this coverage plan.
