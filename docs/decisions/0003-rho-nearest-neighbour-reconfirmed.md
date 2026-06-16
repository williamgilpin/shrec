# 0003 — ρ ≡ minₘ dᵢₘ is *not* a typo (re-verified against the PDF)

**Status:** confirmed · **Date:** 2026-06-10

## Context

The most consequential change in the project's history was the MM5 ρ
off-by-one fix (`dataset_to_simplex` had ρ as the *second*-nearest
neighbour). When auditing for "an appendix typo," the natural suspicion is
that the paper's ρ definition might itself be the typo — i.e. that the code
was "right" and the paper "wrong." This file records the re-verification
that it is **not**, so the fix stands.

## Evidence (PDF p.11, Appendix B, raw extraction)

> "For the ith row of this graph, we find the set of M smallest elements
> `{d^(k)_im}`, which correspond to the distances to the M nearest
> neighbors of point i. The smallest member of this set determines the
> constant: `ρ_i ≡ minₘ {d^(k)_im}`."

This is an **explicit formula** — `ρ_i = the minimum = the nearest
neighbour distance** — not loose prose that could be paraphrasing. A typo
would have to corrupt a typeset `min` operator, which is far less likely
than a dropped scalar coefficient (cf. 0002). All four authorities agree:

1. **paper** — `ρ_i ≡ minₘ dᵢₘ` (nearest);
2. **umap** — `local_connectivity=1` ρ = nearest neighbour (numerically
   verified in `test_recurrence_simplicial.py::test_rho_matches_umap_exactly`);
3. **original author's intent** — pre-refactor comments `# distance to the
   nearest neighbor` and `# drop self`, plus a commented-out *correct*
   `np.sort(dmat)[:, 1:k+1]`;
4. **internal consistency** — ρ = second-nearest matched *no* source.

## Decision

The paper's ρ definition is correct; the historical bug was in the code's
neighbour slicing (`[1:k+1]` after an inf-fill double-skipped to the
second-nearest), already fixed to `[:k]` in `simplicial.py` and pinned by
MM5. No further action; recorded here only to close the "is *this* the
appendix typo?" question with a definitive **no — the typo is the
consensus `1/K`, see 0002.**
