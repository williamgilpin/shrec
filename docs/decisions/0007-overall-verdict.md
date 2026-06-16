# 0007 — Overall verdict: the suite tests the right math

**Status:** accepted · **Date:** 2026-06-10

## Verdict

The `MM<n>` math-correctness suite **tests the correct things**. Every
closed-form oracle audited encodes the paper's actual equation, with the
correct neighbour count, the correct ρ convention (post-fix), and the
correct Laplacian/Fiedler construction. The pipeline now runs the paper's
fuzzy-simplicial consensus (`data_to_connectivity2`) in *both* default
models, so the tests exercise the algorithm the paper describes — not the
old fixed-scale exp kernel.

## Faithful, high-confidence oracles

- **MM1–MM5** — σ defining equation, scale-covariance, unit diagonal,
  fuzzy-union range, umap parity. Literal transcriptions of Appendix B's
  equations; `M = k` neighbour count correct; the umap σ gap correctly
  classified as a *convention*, not a bug (see 0004).
- **MM22 / MM23 / MM33 / MM34 / MM36** — Fiedler eigenvector of `L = D − A`
  on hand-built block and cycle graphs, with the analytically-correct
  targets (block-difference vector; first Fourier harmonic; degenerate
  subspace handled as a subspace claim; NCut generalised eigenproblem;
  eigenvector index contract). Mathematically careful — notably MM22 uses
  cosine *without* mean-subtraction precisely so it can see the
  SVD-vs-Fiedler bug, and MM23 asserts the *subspace* because the Fiedler
  eigenvalue is doubly degenerate.
- **MM13 / MM15** — permutation and per-channel-affine invariances; exact
  symmetry arguments.

## Findings requiring (doc-only) attention

- **0002 — the appendix typo (confirmed).** The consensus equation is
  printed as a bare sum `A_ij ≡ Σ_k A^(k)_ij`; the prose says "elementwise
  average." The code correctly follows the prose. Immaterial downstream.
  Recommend a margin note in `architecture.md`.
- **0005 — MM16 over-claims.** Its assertion (label ARI) is scale-invariant
  and so cannot distinguish mean from sum; the docstring claims it pins the
  `1/K` normalisation, which it does not. Recommend rewording + an optional
  magnitude assertion. No code bug.
- **0006 — MM28 / MM27 provenance.** Legitimate but figure-interpretation /
  calibrated-threshold reproductions, not closed-form oracles. Already
  honestly documented; weight accordingly.

## Honest-modelling calls already made by the suite (endorsed)

- **MM14** downgraded time-reversal from "exact" to "approximate" once
  delay embedding enters and after the sharper post-MM5 kernel surfaced the
  driven logistic map's genuine irreversibility. Correct call: let the test
  track the truth, not the comfortable number.
- **MM20** pinned as a *representational* limit (period-4 collapses to a
  2-way split), proven by an oracle k-means that also fails — not chased as
  a Leiden-resolution artifact. Correct diagnosis.

## Bottom line

No canonical algorithm code change is warranted by this audit. Two
documentation tightenings are recommended (0002 margin note; 0005 MM16
docstring + catalog row). The remembered "appendix typo" is real, located,
and harmless to the results — and the audit additionally surfaced that the
test nominally guarding it (MM16) doesn't actually guard it, which is the
more useful finding of the two.
