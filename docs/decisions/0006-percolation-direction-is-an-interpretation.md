# 0006 — MM28's N-direction is a figure interpretation, not a printed equation

**Status:** accepted with caveat · **Date:** 2026-06-10

## What the paper prints (PDF p.17, Appendix E.3)

> "we extract and measure the aggregated time series adjacency matrix A
> before finding the driver … we compute the order parameter
> `T_LCC/T ∈ [0,1]`, where `T` is the total number of nodes and `T_LCC` is
> the number of nodes comprising the largest connected component. As this
> value approaches one, it indicates the onset of percolation in
> finite-size undirected networks."

The paper's **printed content** is: (a) the *definition* of the order
parameter `T_LCC/T`, and (b) the statement that `→1` means percolation.
It does **not** print an equation for *how `T_LCC/T` scales with `N`*.

## What MM28 asserts

`TestPercolationOrderParameter` asserts `T_LCC/T` **decreases** with the
number of responses `N` (Spearman `< −0.8`, net drop `> 0.3`), on the
simplicial consensus, weak coupling (κ=0.1), absolute threshold θ=0.55.

This direction is taken from **Fig. 6** ("percolation loss precedes
accurate reconstruction"), cited in the docstring — i.e. it is an
*interpretation of a figure*, the soft end of the oracle ladder, not a
transcribed equation. The two construction choices (absolute — not
quantile — threshold; weak coupling) are documented empirical
necessities, not paper-stated conditions.

## Is the interpretation correct?

Yes, and it is internally coherent with the rest of the paper: the
discrete-driver section of Appendix B explicitly notes that "spurious
recurrences cause all nodes to become mutually reachable" — i.e. a fully
percolated graph (`T_LCC/T → 1`) is the *degenerate* case where driver
states are *not* separable. Adding responses lets the consensus vote down
spurious bridges, fragmenting the giant component into driver-state basins,
so `T_LCC/T` falls as `N` grows. That is the desired direction and matches
Fig. 6's narrative.

## Caveat to record

MM28 is a **characterisation / figure-reproduction** test, not a
closed-form oracle. Its assertions (`Spearman < −0.8`, `drop > 0.3`) are
calibrated thresholds on a seed-averaged curve, not paper-stated numbers,
and they depend on construction choices (θ, κ, threshold-type) that the
paper does not fully specify. This is acceptable and already honestly
documented in the test docstring and Round 12 of the learning notes — but
it should not be read as "the paper states `T_LCC/T ∝ f(N)` and we match
it." Its evidentiary weight is lower than MM1–MM5 / MM22–MM23.

## Decision

Keep MM28 as-is; it is a legitimate, well-documented reproduction of the
percolation *phenomenon*. Record here that its link to the paper is a
figure interpretation plus calibrated thresholds, so future readers weight
it accordingly. Same note applies in spirit to MM27 (the Acc-scaling form
*is* a printed equation, p.17, but the fit is on the continuous Spearman
path rather than the paper's discrete ARI — a documented, justified
substitution).
