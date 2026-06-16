# 0004 — σ defining-equation tests (MM1–MM5) are paper-faithful

**Status:** accepted · **Date:** 2026-06-10

## The paper's equation (PDF p.12, Appendix B)

> "We next perform a numerical root-finding calculation to find the value
> of `σ_i` such that
> `log₂ M = Σ_{m=1}^{M} e^{−ReLU(d^(k)_im − ρ_i)/σ_i}`,
> where `ReLU(x) ≡ max(0, x)`. Having calculated `ρ_i, σ_i`, we transform
> the original distance matrix into an affinity matrix via
> `A^(k)_ij = e^{−ReLU(d^(k)_ij − ρ_i)/σ_i}`."

Here `M` is the neighbourhood size — "the set of `M` smallest elements …
the `M` nearest neighbors." The sum runs over those same `M` neighbours,
**including** the nearest one (whose term is `e^{−ReLU(0)/σ} = 1`).

## What the code and tests do

- `simplicial.py:fit_rho_sigma` solves
  `Σ_m exp(−ReLU(d_m − ρ)/σ) = log₂(k)` over the `k` nearest distances
  (`dists[:, :k]`, nearest included). So **`M = k`**, and the nearest-term
  is `1`, exactly as in the paper.
- **MM1** (`TestDefiningEquation`) asserts the residual
  `|Σ exp(...) − log₂ k| < 1e-6` on every row. ✔ This is the literal
  defining equation; it is the test that caught the `fsolve` step-size
  stall.
- **MM2** asserts `(ρ, σ) → (αρ, ασ)` under `d → αd`. ✔ The equation is
  scale-covariant by construction (σ carries units of distance).
- **MM3 / MM4** assert unit diagonal and the `A + Aᵀ − A∘Aᵀ` fuzzy-union
  symmetrisation lands in `[0,1]` and is symmetric. ✔ Matches the
  affinity formula plus the standard t-conorm.
- **MM5** reconciles `fit_rho_sigma` against umap. The *one* residual
  discrepancy with umap — σ differing by ~4% — is a **documented
  convention**, not a bug: umap's `n_neighbors = k` counts the query point
  itself, summing over `k−1` real neighbours toward `log₂(k)`; the
  paper/this code sum over `k` real neighbours. Feeding the solver the
  `k−1` nearest makes σ match umap to ~1e-6, proving the solvers are
  otherwise identical.

## Subtlety worth recording

The paper's `M` and umap's `n_neighbors` count neighbours **differently**
(paper excludes self from the count of real neighbours summed; umap
includes self). The code follows the **paper** (`M = k` real neighbours).
This is the correct choice for fidelity to Appendix B, and MM5's two
assertions cleanly separate "the off-by-one bug (ρ, now fixed)" from "the
self-counting convention (documented, not fixed)."

## Decision

MM1–MM5 encode the paper's σ/ρ/affinity equations faithfully, including
the correct `M = k` neighbour count. No change. The umap self-counting
convention is correctly treated as a convention, not a target to match in
the canonical path.
