# Decision records — math-accuracy audit of the test suite (2026-06-10)

This folder is an append-only log of the key decisions made while auditing
the SHREC math-correctness tests (`MM<n>`, catalogued in
`../tests-math.md`) for **mathematical fidelity to the paper**
(Gilpin, *Phys. Rev. X* **15**, 011005 (2025); local PDF
`../PhysRevX.15.011005.pdf`).

The driving question from the maintainer: *are the tests testing the
correct things?* — with an explicit instruction to double-check a
remembered "potential typo in the appendix."

Each file is one decision/finding, ADR-style (context → finding →
decision → status). Numbered, never renumbered.

| # | Title | Status |
|---|-------|--------|
| [0001](0001-audit-scope-and-method.md) | Audit scope and triangulation method | accepted |
| [0002](0002-consensus-equation-appendix-typo.md) | The appendix typo: consensus equation prints a bare sum, prose says "average" | **confirmed; note applied** |
| [0003](0003-rho-nearest-neighbour-reconfirmed.md) | ρ ≡ minₘ dᵢₘ is *not* a typo — re-verified against the PDF | confirmed |
| [0004](0004-defining-equation-faithful.md) | σ defining-equation tests (MM1–MM5) are paper-faithful | accepted |
| [0005](0005-mm16-overclaims-normalisation.md) | MM16 cannot distinguish mean from sum — docstring over-claims | **applied** |
| [0006](0006-percolation-direction-is-an-interpretation.md) | MM28's N-direction is a figure interpretation, not a printed equation | accepted with caveat |
| [0007](0007-overall-verdict.md) | Overall verdict: the suite tests the right math | accepted |
| [0008](0008-pipeline-modularization.md) | Pipeline modularization: stage strategies + preset models | **done** |
| [0009](0009-pipeline-combo-coverage.md) | Coverage plan for the composable pipeline (tests/benchmarks/examples) | **executed** |

## One-paragraph summary

The suite is **mathematically faithful to the paper**. The remembered
"appendix typo" is real and now pinned down (0002): the consensus-matrix
equation is printed as a bare sum `A_ij ≡ Σ_k A^(k)_ij` while the prose
two words earlier says "elementwise **average**"; the code correctly
follows the prose (divides by the response count). The typo is immaterial
to driver reconstruction because every downstream step (Leiden modularity,
Fiedler eigenvector of `L = D − A`) is invariant to a global positive
scale — which is *also* why the test that nominally guards this (MM16)
cannot actually tell mean from sum (0005). The other equation a reader
might suspect — `ρ_i ≡ minₘ dᵢₘ` — is an explicit formula and **not** a
typo (0003); the historical off-by-one was in the *code*, already fixed.
No code change is required by this audit; two documentation tightenings
are recommended (0005, and a margin note for 0002).
