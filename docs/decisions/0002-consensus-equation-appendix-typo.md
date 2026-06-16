# 0002 — The appendix typo: consensus equation prints a bare sum

**Status:** confirmed; margin note applied (2026-06-16) · **Date:** 2026-06-10

> **Applied 2026-06-16.** The recommended margin note landed in
> `architecture.md` §1 step 4 (the printed-sum typo + the immateriality
> argument), and the stale `aggregation_order` reference in that paragraph was
> corrected to the new `aggregation` modes.

## The remembered typo, located

Appendix B, PDF p.12, defining the consensus affinity matrix. The
sentence reads (raw `pypdf` extraction, page 12):

> "We define a consensus affinity matrix as the elementwise **average**
> across the simplicial complexes for all individual response variables,
> `A_ij ≡ Σ_{k=1}^{K} A^(k)_ij`."

The **prose says "elementwise average"**; the **printed equation is a bare
sum** — there is no `1/K` (or `K⁻¹`) coefficient in front of the
summation. The two disagree by exactly the normalisation factor.

## Why this is a real typo and not a `pypdf` artifact

The raw extracted bytes around the equation are:

```
... all individual response variables, Aij ≡ PK
k¼1 AðkÞ
ij .
```

`PK … k¼1` is `pypdf`'s rendering of `Σ_{k=1}^{K}`. Crucially, **`pypdf`
does render leading fractions elsewhere in this same paper** — e.g. the
random measurement function on p.13 extracts as `pðkÞðxÞ¼ 1=√...`, with
the `1=` (i.e. `1/`) intact. So a `1/K` coefficient, had it been
typeset, would have survived extraction. Its absence is real: the
published equation omits the normalisation that its own prose mandates.

(Verification was limited to text extraction; a pixel render of the
equation was not available in-container — `pymupdf`/poppler absent — but
the calibration against other fractions in the same document makes the
text-level conclusion high-confidence.)

## What the code does — and whether it matters

`recurrence/consensus.py:data_to_connectivity2` accumulates
`dataset_to_simplex(X0, ...) / nb` over the `nb` responses — i.e. a **mean**.
The code follows the **prose** ("average"), which is the intended math, not
the literally-printed sum. So the implementation is correct.

The typo is **immaterial to the reconstructed driver**, because every
downstream consumer of `A` is invariant to a global positive rescaling:

- **Discrete path (Leiden).** Modularity is computed from edge weights
  relative to the total weight; scaling every `A_ij` by `1/K` leaves the
  modularity objective — and therefore the partition — unchanged.
- **Continuous path (Fiedler).** `L = D − A` scales by the same constant,
  so the eigen*vectors* are identical; only the eigen*values* rescale.

So sum and mean produce the *same* driver. The paper's headline results are
unaffected by the misprint; it is a cosmetic error in the equation, caught
only because the prose pins the intent.

## Decision

- **No code change.** The implementation already matches the prose/intent.
- **Record the typo here** so a future reader comparing line-by-line to the
  PDF does not "fix" the code to match the misprinted sum.
- **Recommended (doc-only):** add a one-line margin note in
  `docs/architecture.md` step 4 — "the paper's printed Eq. for `A` omits the
  `1/K`; the prose ('elementwise average') and this code use the mean; the
  two are equivalent downstream up to a global scale."
- See **0005** — this typo is *why* the MM16 idempotence test cannot
  actually distinguish mean from sum.
