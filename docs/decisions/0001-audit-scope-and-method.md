# 0001 — Audit scope and triangulation method

**Status:** accepted · **Date:** 2026-06-10

## Context

The maintainer asked for an audit of *mathematical accuracy* of the
`MM<n>` test suite — not "do the tests pass" (they do: 153 passed / 6
skipped / 1 xfail per `CLAUDE.md`) but "do the assertions encode the
**right** math, the math the paper actually specifies?" — and to
double-check a remembered "potential typo in the appendix."

## What "testing the correct thing" means here

A test can be green and still test the wrong thing in three ways:

1. **Wrong target.** The asserted value/relation does not match the
   paper's equation (e.g. summing over the wrong index, a missing
   normalisation, the wrong neighbour count).
2. **Vacuous guard.** The assertion is satisfied by a *family* of
   implementations wider than the correct one, so it would not fail on
   the bug it claims to catch (see 0005).
3. **Right value, wrong provenance.** The number is defensible but is an
   interpretation of a figure rather than a printed equation, and should
   be labelled as such (see 0006).

## Method

Per the repo's own house rule
(`memory: feedback_verify_before_changing_canonical`), claims about the
canonical algorithm are settled by **triangulating four authorities** and
acting only when they converge:

1. the **primary spec** — Appendix B / E of the PDF, extracted with
   `pypdf` (raw byte inspection, not just rendered text, to catch
   dropped fraction coefficients);
2. the **reference implementation** — `umap.umap_` for the fuzzy
   simplicial set;
3. the **original author's intent** — git archaeology on the pre-refactor
   `models.py` (comments and commented-out lines as intent oracles);
4. **internal consistency** — does a candidate reading match *any* source,
   or is it an artifact with no provenance?

## Surface audited

- `recurrence/simplicial.py`, `recurrence/consensus.py`,
  `recurrence/kernel.py`
- `models/recurrence_clustering.py`, `models/recurrence_manifold.py`
  (confirmed both default models now consume the simplicial consensus
  `data_to_connectivity2`, closing the 2026-05 paper-vs-code path gap)
- the math tests: `test_recurrence_simplicial.py`,
  `test_models_invariances.py`, `test_models_recurrence_manifold.py`,
  `test_scaling_laws.py`, and the catalog `docs/tests-math.md`
- Appendix B (PDF p.11–12) and Appendix E.2/E.3 (PDF p.17) of the paper.

## Decision

Proceed with the four-source method. Record each load-bearing finding as
its own numbered decision file. Recommend documentation changes only;
make **no** changes to canonical algorithm code as a result of this audit
(none is warranted — see 0007).
