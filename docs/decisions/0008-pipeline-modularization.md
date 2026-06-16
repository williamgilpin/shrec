# 0008 — Pipeline modularization: stage strategies + preset models

**Status:** done · **Date:** 2026-06-16
**Supersedes the "where to add new behaviour" sketch in `architecture.md` §5.**

## Context

The audit in 0001–0007 closed the *math* question. This record addresses
the *structural* one the maintainer raised: "implement different aspects of
the toolset as fluidly and flexibly as possible." The May-2026 refactor did
a good **horizontal** split (stage-named sub-packages, free-function
primitives, sklearn-clean base) but left the **vertical composition rigid**:

- Each model hardwires the full pipeline imperatively in `fit()`.
  `RecurrenceClustering.fit` and `RecurrenceManifold.fit` share the
  identical `_preprocess → _make_embedding → data_to_connectivity2` prefix
  (copy-pasted, in neither a base method nor a helper) and diverge only at
  the reconstruction step.
- Extensibility is **inconsistent**: community detection is hot-swappable
  by string (`_leiden(method=...)`), but choosing simplicial-vs-exp
  recurrence, or Leiden-vs-Fiedler reconstruction, means picking a
  different *class*. There is no seam to mix "simplicial recurrence +
  union-find" or "exp-kernel + Fiedler".
- **Core knobs are unreachable.** Both canonical models call
  `data_to_connectivity2(X, time_exclude=...)` and never plumb `k`
  (neighbourhood size) or `tol`. Defaults even disagree across the seam
  (`dataset_to_simplex` k=20 vs `data_to_connectivity2` k=10).
- **Parameter soup.** `RecurrenceModel.__init__` carries ~24 params on the
  shared base; `eps`, `merge`, and `aggregation_order` are **dead**
  (consumed by no model — classical hardcodes `ord=500.`), and `scale`
  applies to the classical baseline only.
- **Welded sub-stages.** Consensus is hardcoded to the mean
  (`data_to_connectivity2`); the distance metric is hardcoded `cdist`
  inside each recurrence primitive despite a `metric="dtw"` claim in the
  base docstring.

## Decision

Introduce two **stage-strategy** seams and recast the four public models as
thin **presets** over a single composable pipeline.

### Stage protocols

```
Connectivity:   (X_embedded: (N, T, D))  ->  A: (T, T)        # recurrence + consensus
Reconstructor:  (A: (T, T))              ->  labels_: (T,) | (T, n_components)
```

### Concrete strategies

- `recurrence/connectivity.py`
  - `SimplicialConnectivity(k, tol, time_exclude, aggregation, metric)` — the
    canonical adaptive fuzzy simplicial complex + consensus.
  - `ExpKernelConnectivity(scale, order, sparsify, tolerance, weighted,
    time_exclude, metric)` — the Sauer exp-of-distance kernel (+ optional
    sparsify). Resurrects the dead `aggregation_order` as `order`, at the
    stage where it belongs.
  - `CommonNeighborsConnectivity(percentile)` — Hirata–Nomura recurrence.
  - `PrecomputedConnectivity(A)` — inject a fixed affinity (testing /
    power-user); replaces the `monkeypatch.setattr(..., data_to_connectivity2)`
    seam the spectral tests used.
- `reconstruct.py`
  - `LeidenReconstructor(resolution, objective, method, random_state)` —
    discrete driver (sets `n_clusters`/`has_unclassified`).
  - `FiedlerReconstructor(n_components, normalize_laplacian, verbose)` —
    continuous driver (Laplacian Fiedler vector + connectivity guard).
  - `UnionFindReconstructor()` — Sauer equivalence classes.
  - `IsomapReconstructor(n_components)` — HN manifold embedding.

### Composition

`models/pipeline.py:ShrecPipeline(RecurrenceModel)` owns the invariant
spine:

```
fit(X): X = preprocess(X); X = embed(X)
        A = connectivity(X);  (store if requested)
        labels_ = reconstructor(A);  set indices + any extra attrs
```

The four public names become presets that map their scalar knobs onto stage
objects (built lazily in `_connectivity_stage()` / `_reconstructor_stage()`,
so `__init__` stays pure attribute-assignment per the sklearn convention).
The public constructor signatures are **unchanged**
(`RecurrenceClustering(resolution=...)`, `RecurrenceManifold(n_components=...,
normalize_laplacian=...)`, etc.), so existing call sites and the back-compat
`models.models` shim keep working.

### Param-soup cleanup

- Remove dead `eps`, `merge`, `aggregation_order` from `RecurrenceModel`.
- Move `scale`/`order` onto the classical preset / `ExpKernelConnectivity`.
- Plumb `k`, `tol`, `metric`, `aggregation` through `SimplicialConnectivity`
  so the central knobs are reachable from the model; reconcile the k=10
  default (the canonical models' effective value is preserved exactly).

## Constraints honoured (so the refactor is behaviour-preserving)

- **Numerics identical** on the default path: `SimplicialConnectivity`
  defaults (`k=10, tol=1e-5, aggregation="mean", metric="euclidean"`)
  reproduce `data_to_connectivity2(X, time_exclude=...)` byte-for-byte; the
  mean aggregator keeps the streaming accumulation.
- **sklearn contract** (MM30/31): presets keep `def __init__(self, <scalars>,
  **kwargs)` (get_params returns the scalar knobs, as today); `__init__`
  only assigns; `fit` returns `self`; no global RNG mutation (MM18).
- **Back-compat shim** `models/models.py` keeps exporting every previously
  importable name (`dataset_to_simplex`, `relu`, `_leiden`, the four models,
  the connectivity fns).
- **Test seam upgrade**: the 3 `monkeypatch.setattr(RM,
  "data_to_connectivity2", ...)` sites in `test_models_recurrence_manifold.py`
  move to `PrecomputedConnectivity` injection — same assertions, cleaner
  seam, and a live demonstration of the new flexibility.

## Refactor plan (ordered, each step test-gated)

1. `recurrence/simplicial.py`, `recurrence/consensus.py`: add `metric`
   (→ `cdist`) and `aggregation` (mean stream / pnorm / max), defaults
   identical. Run `test_recurrence_*`.
2. `recurrence/connectivity.py` (new): the four `Connectivity` strategies.
3. `reconstruct.py` (new): the four `Reconstructor` strategies (lift the
   exact bodies out of the current `fit`s).
4. `models/pipeline.py` (new): `ShrecPipeline`.
5. Recast `recurrence_clustering.py` + `recurrence_manifold.py` as presets.
   Run the manifold/clustering/invariance/driver tests.
6. Recast `classical.py` + `hirata_nomura.py` as presets.
7. `models/base.py`: drop dead params; move `scale`.
8. Update `__init__.py` exports, the `models.models` shim, and the 3
   manifold-test seams.
9. Full suite green (`uv run pytest`). Update `architecture.md` §2/§4/§5
   and `CLAUDE.md` layout.

## Outcome (2026-06-16)

All nine steps landed. Full suite unchanged: **153 passed, 6 skipped, 1
xfailed** (`-m "not slow"`: 150 passed; the 3 slow sweeps MM21/27/28 pass).
Numerics on the default path are byte-identical (the canonical models'
`data_to_connectivity2(X, time_exclude=...)` call is reproduced exactly by
`SimplicialConnectivity`'s defaults). The 3 spectral oracles moved off
`monkeypatch.setattr(..., data_to_connectivity2)` onto
`PrecomputedConnectivity` injection. New public surface: `ShrecPipeline`,
`shrec.recurrence.connectivity.*`, `shrec.reconstruct.*`. Dead base params
(`eps`, `merge`, `aggregation_order`) removed; `scale`/`order` relocated to
the classical preset / `ExpKernelConnectivity`; `k`/`tol`/`aggregation`/
`metric` now reachable from the simplicial presets.

## Consequences

- **Flexibility**: any (connectivity × reconstructor) combination is a
  one-line `ShrecPipeline(...)`; new stages are new strategy classes, not
  new models. The string-dispatch / class-identity asymmetry is gone — every
  stage is a first-class swappable object.
- **Discoverability**: stage params live on the stage they configure; the
  base shrinks to genuinely shared preprocessing/embedding knobs.
- **Cost**: one new concept (stage strategies) and a wider public surface
  (`ShrecPipeline`, `connectivity`, `reconstruct`). The four presets keep
  the familiar names so casual users see no change.
