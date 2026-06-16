# 0005 — MM16 cannot distinguish mean from sum; docstring over-claims

**Status:** recommend doc fix · **Date:** 2026-06-10

## The finding

MM16 (`TestResponseDuplicationIdempotence`,
`tests/test_models_invariances.py`) carries this docstring:

> "Consensus aggregation is a *mean* over responses … Pins that the
> consensus step weights responses uniformly **and renormalises by the
> response count (rather than accumulating raw mass)**."

The assertions are `adjusted_rand_score(labels_one, labels_dup) ≈ 1` (and
`|cos| > 0.999` for the manifold). Both metrics are **invariant to a
global positive rescaling of `A`** (this is exactly the property that
makes the 0002 typo harmless — Leiden modularity and the Fiedler
eigenvector don't see a constant factor).

Therefore: if the code accumulated **raw mass** (a *sum*, `Σ_k A^(k)`,
i.e. the paper's literally-printed equation), duplicating the ensemble
would double `A` — and MM16 would **still pass**, because doubling is a
global scale and the labels are unchanged. The assertion the docstring
claims to make ("renormalises by the count, rather than accumulating raw
mass") is **not** the assertion the test actually performs.

What MM16 genuinely pins is the weaker, still-valuable property:
**duplication-invariance of the driver** — which holds for sum *and* mean
alike. That is a correct and useful regression guard (it would catch, e.g.,
a consensus that weighted responses non-uniformly, or concatenation along
the wrong axis). It simply does not pin the `1/K` normalisation.

## Why this matters / doesn't matter

- It does **not** indicate a code bug: `data_to_connectivity2` does divide
  by `nb`, so the normalisation is present.
- It **does** mean the test catalog slightly overstates coverage: there is
  currently **no** test that would fail if someone changed the consensus
  from mean to sum. Given 0002 (the change is downstream-immaterial), that
  is arguably fine — but the docstring should not claim otherwise.

## Decision / recommendation

1. **Reword the MM16 docstring** to claim what it proves:
   "duplicating the ensemble leaves the driver unchanged (the consensus
   weights responses uniformly; because every downstream step is
   scale-invariant, this holds whether the consensus sums or averages —
   so this test does *not* pin the `1/K` factor itself; see decision
   0002)."
2. **Optional, if pinning the `1/K` is wanted:** add a direct unit
   assertion on `data_to_connectivity2` output *magnitude* — e.g.
   `data_to_connectivity2([X, X]) ≈ data_to_connectivity2([X])`
   elementwise (mean is idempotent under duplication; sum is not). That is
   the only assertion that actually distinguishes the two.
3. Update `docs/tests-math.md` MM16 row to match the reworded claim.

No canonical code change. This is a test-documentation accuracy fix.
