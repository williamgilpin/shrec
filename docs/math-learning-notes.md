# SHREC math — learning dump

> Working scratchpad written *while* building the MM1–MM4 inner-math tests
> and fixing the σ root-solve (branch `fix/simplex-sigma-solve`, 2026-05-28).
> Unstructured on purpose: raw thoughts, "why does this work" moments, and
> **learning topics** flagged for a future Quarto explainer pass. Collate
> later → pick the most human-readable directions for understanding the math.
>
> Convention: a bullet tagged **📚 LEARNING TOPIC** is a candidate Quarto
> page/section. Everything else is context for why it's interesting.

---

## The object under test: the fuzzy simplicial complex (Appendix B, step 3)

For one response channel we have a point cloud (the delay-embedded trajectory).
For each point *i* we build a "fuzzy" membership to every other point *j*:

```
a_ij = exp( -ReLU(d_ij - ρ_i) / σ_i )
```

- `ρ_i` = distance to the *nearest* neighbour of *i*. Subtracting it and
  ReLU-clipping means the nearest neighbour gets membership `exp(0)=1`, and
  anything closer than ρ (only *i* itself) is also clamped to 1. This is the
  "local connectivity = 1" idea: every point is certainly connected to its
  closest neighbour, regardless of absolute scale.
- `σ_i` = a per-point bandwidth chosen so the *row* carries a fixed amount of
  total fuzzy mass.

The bandwidth is fixed by the **defining equation**:

```
Σ_m exp( -ReLU(d_im - ρ_i) / σ_i )  =  log₂(k)
```

summed over the *k* nearest neighbours.

- **📚 LEARNING TOPIC — "Why log₂(k)?"** This is the UMAP `smooth_knn_dist`
  target. It's a *perplexity-like* normalisation: instead of every point
  having k hard neighbours, each point has a fixed *effective* number of
  fuzzy neighbours = log₂(k). It makes the per-point bandwidth adapt to local
  density (dense region → small σ, sparse region → large σ) while keeping the
  total "information" per row constant. Worth a side-by-side with t-SNE's
  perplexity (which uses Shannon entropy / log, not log₂). Good Quarto figure:
  same point cloud, color each point by its solved σ_i, show σ tracking local
  spacing.

- **📚 LEARNING TOPIC — "Local connectivity and the ρ shift."** Why subtract
  the nearest-neighbour distance at all? Show what the affinity graph looks
  like with vs without the ρ shift on a cloud with two density scales. Without
  it, the sparse cluster fragments. This is *the* reason UMAP/SHREC handle
  multi-scale data.

---

## The bug this branch fixes (and what it taught me)

The σ in the defining equation has to be found numerically — no closed form.
The original code used `scipy.optimize.fsolve` with the Newton-style MINPACK
`hybrd` routine, starting from `x0 = ρ_i`.

**What went wrong:** on ~6% of rows `fsolve` returned `ier=1` ("converged")
but the residual of the defining equation was ≈ 2.8, not < 1e-6 — and σ came
back *exactly equal to the initial guess ρ_i*. It never took a step.

- **📚 LEARNING TOPIC — "Convergence criteria lie."** MINPACK's `hybrd`
  declares success on the *step size* falling below `xtol`, NOT on the
  residual being small. If the first proposed step is tiny (bad scaling of
  the analytic Jacobian here), it can quit at iteration 0 and still report
  success. Great cautionary tale: *always check the residual yourself*, never
  trust the solver's status flag. This is a general numerical-methods lesson,
  not SHREC-specific — strong candidate for a standalone explainer.

**Why the fix is clean — monotonicity.** The left side of the defining
equation is *strictly monotone increasing* in σ:

- as σ → 0⁺, every non-nearest term `exp(-positive/σ) → 0`, so the sum → (number
  of neighbours exactly at distance ρ) = 1 in general → and `1 < log₂(k)` for
  k > 2.
- as σ → ∞, every term `exp(-x/σ) → 1`, so the sum → k > log₂(k).

A continuous function going from below-target to above-target has exactly one
root, and a *bracketing* method (`scipy.optimize.brentq` on `[1e-12, 1e6]`)
finds it to machine precision (residual ~1e-12 on every row). No initial guess,
no Jacobian, no silent stalls.

- **📚 LEARNING TOPIC — "Bracketing vs Newton root-finding."** When you can
  prove monotonicity + a sign change, bracketing (bisection/Brent) is
  unconditionally robust; Newton/secant are faster but can stall or diverge.
  The SHREC σ-solve is a perfect worked example because monotonicity is easy
  to see from the physics of the kernel. Quarto: plot `f(σ)` for a stalled row,
  mark the fsolve stall point and the true brentq root.

---

## What the four MM tests actually pin (and the math idea behind each)

- **MM1 — defining equation residual.** The direct check: solve, plug back in,
  assert `|Σ exp(...) − log₂k| < 1e-6`. This is the test that *catches the
  fsolve stall*. Lesson: the cheapest possible test (does the equation we
  claim to solve actually hold?) was the one missing, and it caught a real bug.

- **MM2 — scale invariance of (ρ, σ).** If you scale all distances by α, then
  `ρ → αρ` and `σ → ασ`, leaving every affinity `exp(-ReLU(αd-αρ)/(ασ))`
  unchanged. **📚 LEARNING TOPIC — "Dimensional analysis as a test oracle."**
  σ has units of distance; the equation is scale-covariant by construction.
  Recognising the symmetry *before* coding gives you a free, exact test. This
  is a reusable habit: find the equivariance, turn it into an assertion.

- **MM3 — unit diagonal.** `d_ii = 0`, so `ReLU(0 − ρ_i) = 0`, `exp(0) = 1`;
  the fuzzy union `1 + 1 − 1·1 = 1`. A point is fully self-connected. Trivial,
  but pins that the symmetrisation didn't accidentally hollow the diagonal.

- **MM4 — fuzzy-union stays in [0,1] and is symmetric.** The symmetrisation
  `A + Aᵀ − A∘Aᵀ` is the **probabilistic t-conorm** (fuzzy OR):
  `P(i~j) = P(a) + P(b) − P(a)P(b)`, i.e. "edge exists if *either* direction
  votes for it." **📚 LEARNING TOPIC — "Fuzzy set unions / t-conorms."** Why
  this specific formula and not `max` or averaging? It's the probability that
  at least one of two independent directed edges fires. Connects SHREC's graph
  to probabilistic-graph / fuzzy-logic foundations. Nice Quarto: the three
  candidate symmetrisations (max, mean, t-conorm) and how the consensus graph
  differs.

---

## Threads to pull later (broader than this branch)

- **📚 LEARNING TOPIC — "From recurrence to a graph Laplacian."** The
  continuous-driver path takes the Fiedler eigenvector of `L = D − A`, not the
  2nd singular vector of `A` (MM22 pins this). On *irregular* graphs these
  differ. A Quarto page deriving why the Fiedler vector is the natural
  "smoothest non-trivial coordinate" on the recurrence graph would tie the
  whole continuous side together.
- **📚 LEARNING TOPIC — "Consensus over channels = mean of fuzzy graphs."**
  Step 4 averages the per-channel affinity matrices. Why mean and not a fuzzy
  intersection? What does the mean assume about shared-driver structure?
- **📚 LEARNING TOPIC — "The Sauer limit."** Period-2 recovers ARI≈1 but
  period-4 collapses to 2 Leiden communities (MM20 xfail). The math of *why
  community-detection resolution interacts with the driver's period* is a
  genuinely interesting open thread — ties to modularity-vs-CPM objectives.

---

## Late addition — the bug the *fix* exposed (tied neighbourhoods)

Swapping `fsolve → brentq` immediately broke 5 previously-green pipeline tests,
all with `ValueError: f(a) and f(b) must have different signs`. Tracking it down
was itself the best learning moment of the session:

On the **real** driver-response data (logistic maps with `np.clip(·, 0, 1)`),
many embedded points are *exactly identical* because the clip saturates the
series to 0.0 / 1.0. So a point can have all `k` nearest neighbours at the
**same** distance ρ. Then every kernel term is `exp(-ReLU(0)/σ) = 1` for *any*
σ, the row mass is pinned at `k`, and the target `log₂k < k` is **unreachable** —
the defining equation has no solution. `brentq` correctly refuses (no sign
change); `fsolve` had been silently returning the stalled guess and nobody knew.

- **📚 LEARNING TOPIC — "When the equation has no solution."** A whole genre of
  numerical bug: the solver isn't wrong, the *model* is degenerate on this
  input. The honest fix is to detect the degeneracy (here: `f(σ→0⁺) ≥ 0`) and
  define the limiting behaviour explicitly (σ indeterminate → fall back to ρ),
  not to loosen tolerances until it stops complaining. Great Quarto narrative:
  show the constant `f(σ) = k − log₂k` curve for a tied row next to a healthy
  monotone-crossing curve.
- **📚 LEARNING TOPIC — "Quantisation / clipping creates exact ties."** The
  data-generation detail (`clip`) has a downstream geometric consequence
  (duplicate points → distance ties → degenerate local connectivity). Worth a
  short note on how preprocessing choices silently shape the recurrence graph.
- **Meta-lesson for the test suite:** the *unit* test (MM1) and the *pipeline*
  tests disagreed about what "correct" means until the degenerate case was
  handled. Both were needed — MM1 to find the stall, the pipeline tests to
  reveal that the naive fix changed real behaviour. This is the argument for
  keeping both altitudes of test, not just one.

---

# Round 2 — devising parallel & perpendicular tests (2026-05-28)

After MM1–MM4 landed, surveyed the rest of the code surface + complementary
literature (UMAP `smooth_knn_dist`, von Luxburg spectral-clustering tutorial,
Sauer/Takens embedding, the SHREC paper Appendix B itself). Two axes:

- **Parallel** = more of the same family (closed-form inner-math oracles) on
  operators that currently have *none*.
- **Perpendicular** = different *methodology*: property-based, metamorphic,
  differential-vs-reference, fuzz/stability, statistical/surrogate.

## Parallel candidates (closed-form, cheap)

- **MM6** — `kernel.data_to_connectivity` p-norm limits: `ord=1` ⇒ elementwise
  mean of per-channel kernels; `ord→∞` ⇒ min-over-channels distance (Sauer
  `inf_k d^(k)`). Verified no NaN at ord=500; limit shape is assertable.
- **MM7/MM8** — `cdist` isometry invariance (random orthogonal Q + translation)
  and triangle inequality. Trivially true for euclidean, but pins that no code
  path rescales distances asymmetrically.
- **MM9** — `distance_to_connectivity(sparsity=…)` actually hits the requested
  sparsity, and is monotone in the scale.
- **MM10** — `_leiden` backend agreement (graspologic vs igraph vs leidenalg)
  on the barbell graph: ARI=1 across backends.
- **MM11** — `unionfind.DisjointSet` parity with
  `scipy.cluster.hierarchy.DisjointSet` on random merge lists.
- **NEW — Laplacian algebra**: `L = D−A` has row sums 0 (constant vector is
  eigenvector at λ=0); eigenvalues ≥ 0; λ₂ > 0 **iff** the graph is connected.
  This is the oracle for the disconnected-graph bug below.
- **NEW — embedding reconstruction**: `embed_ts`/`hankel_matrix` output equals a
  hand-rolled delay stack `[x_t, x_{t-τ}, …]`; padding modes preserve length.

## Perpendicular candidates (different methodology)

- **📚 LEARNING TOPIC — property-based testing (Hypothesis).** `hypothesis` is
  already a dev dep and registered in pytest but used *nowhere*. Strong fit
  for the algebraic invariants: for arbitrary small clouds, assert
  `dataset_to_simplex` output is symmetric/in [0,1]/unit-diagonal; assert
  `fit_rho_sigma` residual < tol OR the tied-degenerate fallback fired. Lets
  the machine search for the adversarial inputs we'd never hand-pick (it would
  have found the tied-neighbourhood case on its own).
- **📚 LEARNING TOPIC — metamorphic testing.** When there's no ground-truth
  output, test *relations between* outputs: monotonicity (more coupling ⇒
  higher recovery ARI), graceful degradation (ARI decreases monotonically as
  noise σ rises), adding a pure-noise channel must not *improve* a clean
  recovery. These encode the paper's qualitative claims as ordering assertions.
- **📚 LEARNING TOPIC — differential / reference-oracle testing.** Pin our
  operators against an independent implementation: `dataset_to_simplex` vs
  `umap.umap_.fuzzy_simplicial_set` (this is MM5, currently xfail — worth
  *reconciling conventions* rather than leaving frozen); `RecurrenceManifold`
  Fiedler vs `sklearn.manifold.SpectralEmbedding`; `_leiden` vs raw backends.
- **📚 LEARNING TOPIC — surrogate / null-hypothesis testing.** Borrowed from
  nonlinear-time-series practice (the paper itself uses surrogate ideas):
  feed SHREC *independent* responses with **no** shared driver and assert the
  recovered driver is statistically indistinguishable from noise (low ARI /
  flat eigenvector). Guards against the method hallucinating structure — the
  most important property a causal-discovery tool can have.
- **📚 LEARNING TOPIC — numerical fuzz / stability.** Sweep dtype (float32 vs
  64), tiny/huge distance scales, near-duplicate rows, NaN injection. The σ
  bug was exactly a stability failure; a fuzz harness generalises the guard.

## Potential issues found while surveying (grounded, not speculative)

1. **Unnormalised Laplacian degree bias.** `RecurrenceManifold` uses `L = D−A`
   (RatioCut), not `L_sym`/`L_rw` (NCut). On degree-heterogeneous consensus
   graphs the Fiedler vector localises on high-degree nodes (von Luxburg). The
   paper itself flags this: "correct for response bias … might require
   preconditioning the graph Laplacian." → metamorphic test: skewing one
   channel's recurrence density should not dominate the recovered driver.
   **📚 LEARNING TOPIC — RatioCut vs NCut and why normalisation matters.**
2. **Disconnected-graph degeneracy (demonstrated).** With no bridge between
   blocks, λ₂ = 0 (2-dim null space) and `eigh(subset_by_index=[1,1])` returns
   an arbitrary component *indicator* (step function {0, 0.183}), not a smooth
   driver. No connectivity check exists. → guard: assert/ warn when λ₂ ≈ 0;
   test on a deliberately disconnected affinity.
3. **`distance_to_connectivity` bracket fragility — twin of the σ bug.**
   `root_scalar(optfun, bracket=[1e-16, dscale])` assumes a sign change that
   isn't guaranteed for every distance distribution; same failure class we
   just fixed in `fit_rho_sigma`. Needs the same bracket-guard discipline.
4. **igraph backend silently ignores `resolution`.** `communities._leiden`
   igraph branch hardcodes `resolution_parameter=1.0` instead of forwarding
   `resolution`. A backend-agreement test (MM10) that varies resolution would
   catch this. (graspologic path does forward it correctly.)
5. **`subset_by_index=[1, n_components]` is inclusive-count-off-by-one-prone.**
   For `n_components=1` it returns one vector (good), but the semantics ("index
   1 through n_components") differ from "n_components vectors" for >1. Worth a
   shape test pinning the intended contract.

---

# Round 3 — theory deep-dives (2026-05-29)

Educational write-ups attached to the three fixes (igraph resolution, the
connectivity guard, the Hypothesis suite). These are deliberately fuller than
scratch notes — they're the seed text for Quarto explainer pages. Each is
self-contained; trim/expand when collating.

## A. The graph Laplacian and algebraic connectivity (behind the MM33 guard)

For an undirected weighted graph with affinity `A` and degree `D = diag(Σ_j A_ij)`,
the **combinatorial Laplacian** is `L = D − A`. Three facts make it the natural
operator for the continuous driver:

1. **`L` is symmetric PSD.** For any vector `f`,
   `fᵀ L f = ½ Σ_{i,j} A_ij (f_i − f_j)²  ≥ 0`.
   This quadratic form is the "energy" of `f` on the graph — small when `f`
   varies slowly across strongly-connected nodes. So the *smallest* non-trivial
   eigenvector is the *smoothest non-constant coordinate*: exactly what a slowly
   varying shared driver should look like on the recurrence graph.
2. **The constant vector is always an eigenvector at λ₁ = 0** (`L·1 = 0`,
   because rows sum to zero). That's why `RecurrenceManifold` asks `eigh` for
   `subset_by_index=[1, …]` — it *skips* index 0, the uninformative constant.
3. **Algebraic connectivity.** The second-smallest eigenvalue λ₂ (the *Fiedler
   value*) is `> 0` **iff** the graph is connected. More strongly: *the
   multiplicity of the eigenvalue 0 equals the number of connected components*,
   and the null-space is spanned by the component-indicator vectors.

Fact 3 is the entire content of the MM33 bug. When the consensus graph splits
into (nearly) disconnected pieces, λ₂ → 0 and the "Fiedler" eigenvector we pull
out is an arbitrary mixture of component indicators — a *step function*, not a
driver. `eigh` returns it without complaint. The guard watches λ₂ against the
spectral scale (`trace L = Σ degree`) and warns.

- **📚 LEARNING TOPIC — "The Laplacian quadratic form."** Derive
  `fᵀLf = ½ Σ A_ij (f_i−f_j)²` from scratch; it's three lines and it demystifies
  why spectral methods "find smooth coordinates." Best single figure in the
  whole topic: a path graph with its first few Laplacian eigenvectors drawn as
  standing waves (they're literally `cos(πki/n)` — discrete vibration modes).
- **📚 LEARNING TOPIC — "Eigenvalue 0 counts components."** Worked 2-block
  example with the bridge weight swept from 1 → 0, watching λ₂ slide to 0 and
  the eigenvector morph from a smooth ramp into a hard step. This *is* the MM22
  → MM33 continuum in one animation.

## B. RatioCut vs NCut — why `L=D−A` is not the whole story

Minimising `fᵀLf` subject to `f ⟂ 1`, `‖f‖=1` is a relaxation of **RatioCut**
(balance by *number of nodes* per side). The normalised Laplacians
`L_sym = D^{-1/2} L D^{-1/2}` and `L_rw = D^{-1} L` relax **NCut** (balance by
*volume* = total degree per side). The practical difference: with the
unnormalised `L`, a few very high-degree nodes dominate the Fiedler vector, so
the embedding can fixate on the densest response channel rather than the shared
driver. von Luxburg's tutorial recommends normalised Laplacians by default, and
the SHREC paper itself lists "preconditioning the graph Laplacian" as the fix
for response bias. We did **not** change the operator (that would shift the
paper-faithful behaviour); MM33 just makes the degenerate end visible.

- **📚 LEARNING TOPIC — "RatioCut vs NCut, and degree bias."** Side-by-side
  Fiedler vectors of `L` vs `L_rw` on a graph with one inflated-degree cluster.
  Tie it back to the SHREC "response bias" remark — a concrete reason a user
  might want a normalised variant as an option.

## C. Modularity, CPM, and the resolution parameter (behind the igraph fix)

Leiden optimises a **quality function** over partitions. Two choices:

- **Modularity** compares within-community edge weight to what you'd expect in a
  degree-preserving random graph. It has a famous **resolution limit**: below a
  size set by the *total* graph weight, it cannot resolve small communities and
  *merges* them. This is precisely the MM20 xfail — the period-4 driver's four
  states collapse into two modularity communities.
- **CPM (Constant Potts Model)** compares within-community density to a fixed
  threshold `γ` (the resolution). No resolution limit, and `γ` has a direct
  reading: communities are denser than `γ`, sparser between. Sweeping `γ` traces
  the whole hierarchy from one blob (γ→0) to all singletons (γ→1).

The igraph bug hardcoded the resolution to 1.0, so `γ` was silently ignored — a
user sweeping resolution to escape the modularity resolution limit would see *no
change*. MM20 (period-4) is the standing motivation: the documented path to
closing it is a resolution/CPM sweep, which the bug would have quietly defeated.

- **📚 LEARNING TOPIC — "The modularity resolution limit."** The cleanest demo
  of why a hyperparameter-free method can still miss structure: a ring of
  cliques where modularity provably merges adjacent cliques once there are
  enough of them. Direct line to why SHREC's period-4 case is an xfail and what
  CPM buys you.

## D. Why property-based + metamorphic testing fits this codebase

Most SHREC operators have *no closed-form output* on real inputs — you can't
write down the "right" affinity matrix for a logistic ensemble. Two testing
philosophies handle that without a ground truth:

- **Property-based (Hypothesis).** Don't assert the output value; assert
  *properties that must hold for every input* (symmetry, [0,1] range, unit
  diagonal, residual-or-fallback) and let the engine search the input space —
  including shrinking any failure to a minimal counterexample. It explores the
  adversarial corners (duplicate points, collinear clouds, extreme scales) that
  a human fixture-writer would never enumerate. Our `_point_clouds` strategy is
  the reusable template: bound the values, `assume` away the genuinely
  ill-posed inputs, keep `max_examples` modest because each example runs the
  per-row solver.
- **Metamorphic.** Assert *relations between* outputs of related inputs:
  permutation invariance (already MM13), scale invariance (MM2), and — not yet
  built — monotone degradation (more noise ⇒ lower ARI) and the null-input
  property (no shared driver ⇒ no recovered structure). These encode the
  paper's qualitative claims as inequalities, which is often all the science
  actually asserts.

- **📚 LEARNING TOPIC — "Testing without an oracle."** A short methodology page:
  the ladder from exact oracle (MM1) → invariance/metamorphic (MM2, MM13) →
  reference-implementation differential (MM5) → statistical/surrogate (null
  driver). SHREC has a clean example at every rung — an unusually good teaching
  vehicle for scientific-software testing in general.

---

# Round 4 — closing the survey items (2026-05-29)

Took on the three remaining survey issues (#1 unnormalised Laplacian, #3
`distance_to_connectivity` bracket, #5 eigenvector shape). Notes + a couple of
honest negative results worth teaching from.

## #5 was a false alarm — and that's a lesson too

`scipy.linalg.eigh(L, subset_by_index=[1, n_components])` returns indices
`1..n_components` *inclusive* = exactly `n_components` non-trivial eigenvectors.
It was already correct. I'd flagged it as "off-by-one-prone." The right
response to a *suspected* bug that turns out fine isn't to silently move on —
it's to lock the current contract with a test (MM36) so the suspicion can't
resurface and so a real future off-by-one is caught.

- **📚 LEARNING TOPIC — "Inclusive vs half-open ranges in numerical APIs."**
  scipy's `subset_by_index` is inclusive on both ends; NumPy slicing is
  half-open; LAPACK's `il`/`iu` are 1-based inclusive. A short reference table
  of who-means-what would prevent a whole genus of off-by-one bugs.

## #3 — the diagonal floor makes some sparsity targets *infeasible*

`distance_to_connectivity(sparsity=s)` solves for a kernel scale so the mean
affinity equals `1−s`. But `cdist(X, X)` has a **zero diagonal**, and
`exp(-0/x) = 1` for any `x`, so the `N` diagonal ones contribute a fixed
`N/N² = 1/N` to the mean. The mean affinity therefore can never drop below
`1/N`, so **any `s > 1 − 1/N` is unsatisfiable** — and for `N = 8`, `s = 0.99`
needs mean `0.01 < 0.125 = 1/8`. The old `root_scalar` just crashed with the
same "f(a) and f(b) must have different signs" we saw in the σ-solve. The fix
detects infeasibility (`optfun(σ→0) ≥ 0`), warns, and returns the sharpest
kernel; otherwise it expands the upper bracket until it provably contains the
root (monotonicity guarantees one).

- **📚 LEARNING TOPIC — "Feasibility before optimisation."** A recurring
  scientific-computing bug: solving for a target without first asking whether
  the target lies in the operator's range. The diagonal floor here is a clean,
  countable example — you can *derive* the infeasible region (`s > 1 − 1/N`)
  before running any solver. Pairs naturally with the σ-solve's tied-
  neighbourhood degeneracy: both are "the equation has no solution" bugs.

## #1 — degree bias, and an honest negative result

Added `normalize_laplacian` (default off, paper-faithful). The teaching moment
was the *experiment design*: I tried hard to build a graph where unnormalised
spectral clustering **fails** the 2-way split and normalised succeeds — and on
clean, balanced block structures, **both recover the split (ARI = 1)**. The
degree bias does not show up in the *sign* of the Fiedler vector; it shows up
in its *values*: on a degree-heterogeneous graph the unnormalised vector put
~50% of its energy on the high-degree block, the normalised one ~0.1%. Since
SHREC's continuous driver uses the eigenvector *values* (correlated against the
true driver), not a binary cut, that value-level distortion is exactly what
matters — but it's subtle enough that a naive ARI test would have shown "no
difference" and hidden the whole point.

- **📚 LEARNING TOPIC — "When the bug is in the values, not the labels."**
  Spectral methods used for *embedding* (continuous coordinates) have different
  failure modes than the same methods used for *clustering* (discrete labels).
  RatioCut vs NCut barely changes the sign pattern on clean blocks but
  substantially changes the coordinate. Great cautionary tale about choosing a
  test metric that can actually see the failure you care about.
- **📚 LEARNING TOPIC — "Negative results are results."** Document the cases
  where the "obvious" fix shows no benefit on the obvious test. It stops the
  next person re-litigating it and sharpens *why* the real effect is where it
  is (here: values, not labels; continuous driver, not discrete).

---

# Round 5 — the deferred must-tests MM6 & MM27 (2026-05-29)

Closed the last two deferred "must" tests. With these, every must-test is green
or a documented xfail — the algorithm is "green" by the catalog's own bar.

## MM6 — the ensemble aggregation is a power-mean (and floats bite at the limit)

`data_to_connectivity` aggregates per-channel kernels `a_i = exp(-surprise_i/thresh)`
across the ensemble as

    bd = (1/nb · Σ_i a_i**ord) ** (1/ord)   — a power-mean (generalised mean) M_ord.

The two limits the classical baseline relies on fall straight out of power-mean
theory: `M_1` = arithmetic mean; `M_∞` = max. Since `a_i = exp(-surprise_i/thresh)`,
the max over channels is `exp(-(min_i surprise_i)/thresh)` — the *closest-channel*
recurrence, which is exactly the Sauer `inf_k d^(k)` the `ClassicalRecurrenceClustering`
approximates with `ord=500`.

Two things worth teaching here:

- **📚 LEARNING TOPIC — "Power means interpolate min↔mean↔max."** `M_{-∞}=min`,
  `M_0=geometric mean`, `M_1=arithmetic`, `M_2=RMS`, `M_∞=max`, monotone in the
  exponent. SHREC's `ord` is literally this knob: it dials the consensus from
  "average channel" toward "most-recurrent channel." A single figure of `M_p`
  vs `p` on a handful of values makes the whole classical-vs-canonical design
  legible.
- **📚 LEARNING TOPIC — "Why `ord=500` and not `∞`: floating-point reach."**
  The L∞ limit is exact in real arithmetic but `a_i**500` *underflows to 0*
  once `a_i ≲ 0.25` (since `0.25**500 ≈ 10^-301` nears the double floor). So at
  the Sauer setting, low-affinity pairs collapse to 0 — harmless for a
  thresholded graph, but it means the test can only check the max-limit on
  well-conditioned (high-affinity) entries. Convergence is also only
  `O(nb^{-1/ord})`, i.e. *slow*. Good worked example of "the math says ∞, the
  hardware says ~500."

## MM27 — a scaling law, and choosing a regime where the effect exists

The paper claims accuracy grows and saturates with total data:
`Acc(NT/τ) = Acc_max(1 − exp(−β√(NT/τ)))`. Reproducing it taught the most about
*experiment design*, not code:

1. **Pick the observable that can see the effect.** The catalog phrased MM27 in
   discrete-ARI terms, but the period-4 ARI is capped at ≈0.5 by the Leiden
   resolution collapse (MM20). Switching to the *continuous* driver
   (RecurrenceManifold + Spearman |ρ|) gives a smooth accuracy in [0,1] that a
   2-parameter curve can actually be fit to.
2. **Pick a regime where more data helps.** With a clean continuous driver,
   accuracy *already saturates at N=2* — flat curve, nothing to fit. The scaling
   only appears when each response is mildly unreliable, so consensus across
   responses buys something. Light observation noise (σ=0.05) is the sweet spot:
   accuracy climbs 0.29 → 0.75 over N=2→8 then plateaus. σ=0.2 is *too* much —
   it destroys the recurrence structure and accuracy collapses to ~0 for all N.
   There's a genuine signal-to-noise window in which the law is visible.
3. **Assert the claim, not the fit.** R² of the saturating form was ~0.77 —
   good, but seed-dependent. The robust assertions are the *content* of the law:
   β>0 (accuracy increases with NT/τ), Acc_max∈[0.6,1] (it works at saturation),
   a clear small-N→large-N gain, and that the curve beats a flat-mean baseline
   (R²>0.5). Pinning R²>0.9 would be a flaky test masquerading as a precise one.

- **📚 LEARNING TOPIC — "Signal-to-noise windows in inference methods."** Plot
  accuracy vs (noise σ, N) as a heatmap: too little noise → no N-dependence to
  study; too much → no recovery at any N; a diagonal band where consensus pays
  off. This is the percolation/“glass-like” story of the paper made concrete and
  is probably the single most illuminating figure for a methods reader.
- **📚 LEARNING TOPIC — "Testing a scaling law without overfitting it."** The
  general recipe: average over seeds, fit the claimed form, then assert its
  qualitative parameters + a baseline-beating goodness rather than a tight R².
  Distinguishes "the law holds" from "these particular numbers recurred."

---

# Round 6 — investigating MM20 (period-4), and a negative result that matters (2026-05-29)

The remaining xfail MM20 (period-4 driver → ARI≈0.5) carried a hopeful note:
"closing requires Leiden resolution tuning or CPM." Investigated it properly
and the note was **wrong** — closing it requires neither, because the problem
isn't in the clustering step at all. The investigation is a small case study in
*localising a failure to the right stage of a pipeline*.

The pipeline is: embed → per-channel recurrence → consensus graph → community
detection. The ARI≈0.5 could live in any stage. Three probes pinned it:

1. **Resolution sweep (blames the clustering objective).** Modularity at
   res∈{0.5,1,2,4,…} jumps from 1 community (res 0.5) to 2 (res 1) straight to
   ~1000 singletons (res 2) — there is no resolution that yields a stable 4.
   So *if* the structure were present, modularity still couldn't tune to it —
   but this alone doesn't prove the structure is absent.
2. **Oracle spectral k-means=4 (bypasses Leiden entirely).** Take the consensus
   Laplacian's top eigenvectors and k-means them into exactly 4 — the most
   generous clustering possible. Still ARI≈0.5, robustly, across N∈[20,100] and
   coupling∈[0.5,1]. This *removes* the clustering objective as the suspect:
   the eigenvectors themselves don't carry a 4-way split.
3. **Continuous reconstruction (bypasses clustering altogether).** The
   RecurrenceManifold Fiedler vector vs the ordered driver gives Spearman
   |ρ|≈0.38. Even treating period-4 as a continuous ordering problem fails.

Conclusion: the four levels {0.1,0.4,0.6,0.9} collapse to a low/high **2-way**
split *in the recurrence graph itself*. The information loss is upstream of
clustering — in how binary recurrence (co-location in delay space) represents a
4-level driver whose adjacent levels share recurrence basins. No clustering
choice can recover what the graph doesn't encode.

- **📚 LEARNING TOPIC — "Localising failure in a multi-stage pipeline."** The
  general move: replace each downstream stage with an *oracle* (here, k-means
  with the true k; the continuous embedding with no clustering) and see if the
  metric recovers. The first oracle that *doesn't* rescue accuracy contains the
  bottleneck. A clean, reusable debugging discipline worth its own page.
- **📚 LEARNING TOPIC — "What binary recurrence can and cannot encode."** Why a
  2-level (period-2) driver is recovered near-perfectly but a 4-level one
  collapses: recurrence is an equivalence relation (same state ↔ edge), and it
  resolves driver levels only as well as the responses' delay-embedded states
  separate those levels. Adjacent levels with overlapping basins merge. This is
  the conceptual heart of the method's resolution limit and pairs naturally with
  the MM27 signal-to-noise story.
- **Process lesson:** the codebase's xfail note encoded a *plausible but
  untested* cause ("tune Leiden"). Investigating turned a guess into a measured
  fact and saved the next person a fruitless tuning PR. The companion
  characterisation test `test_period_four_is_not_separable_in_graph` now *pins*
  the real cause, and will fire if a future representation change actually fixes
  it. Encoding "why this is hard" as a passing test is as valuable as testing
  "this works."

---

# Round 7 — closing the closed-form inner math: MM7, MM8, MM9, MM11 (2026-06-01)

First batch of the post-PR follow-up. These four are the "boring" oracles — the
distance primitive, the sparsifier, the union-find — but they're the foundation
the headline algorithm stands on, and writing them surfaced a genuine subtlety
in the σ-solver's *tolerance semantics* (see the steep-root note below).

## MM7 — recurrence is invariant to where you stand, not just how far you zoom

The scale-invariance oracle MM2 says: multiply every distance by α and the
affinity is unchanged. MM7 is its rigid-motion sibling: **rotate, reflect, and
translate the whole point cloud and the affinity is unchanged.** Together they
say recurrence depends only on the *intrinsic shape* of the trajectory in delay
space — not on the coordinate frame, the origin, or the units.

The decision worth recording: the catalog originally specced MM7 as "`cdist`
isometry invariance," i.e. test that `scipy`'s distance function is an isometry.
But re-testing a library function you trust is near-zero-value. I *lifted* the
test up a level: assert the invariance of `dataset_to_simplex` itself — the real
ρ/σ pipeline. That actually exercises our code (the σ root-solve, the
symmetrisation) and would catch a frame-dependent bug we could plausibly
introduce; the raw-`cdist` check is kept only as a one-line "here's the geometric
fact it rests on" companion.

- **📚 LEARNING TOPIC — "The two invariances of a metric method: similarity vs
  congruence."** MM2 (scaling) + MM7 (rigid motion) together are exactly the
  group of *similarity transformations*. A method invariant to both is blind to
  the choice of ruler and the choice of frame — it sees only angles and distance
  *ratios*. Worth a short page connecting this to why standardisation
  (z-scoring each response) is the *only* extra normalisation SHREC needs: it
  removes the per-response scale, and rigid-motion invariance is already free.
- **📚 LEARNING TOPIC — "Test the code, not the library."** A reusable rule for
  oracle design: if an invariant is a property of a dependency you trust, push
  the test up until it exercises *your* transformation of that dependency's
  output. The same oracle then doubles as a regression guard for your pipeline.

## MM8 — why "is it a metric?" is a load-bearing question, not pedantry

MM8 pins the triangle inequality (plus zero-diagonal, symmetry, non-negativity)
on the Euclidean `cdist`. Trivially true for Euclidean — so why bother? Because
the *next* stage assumes it. `fit_rho_sigma` sets ρ = the nearest-neighbour
distance and treats `ReLU(d − ρ)` as a non-negative "excess distance" decaying
under a positive bandwidth σ. That story only makes sense if d is a genuine
metric: ρ must be the *minimum* (zero self-distance, non-negativity), and the
local-connectivity argument leans on distances composing sanely. The test is a
**contract pin**: it documents the assumption so that if someone later swaps in
a non-metric (cosine *dissimilarity*, a learned embedding distance, DTW — which
famously violates the triangle inequality) the breakage is caught at the
primitive, not three stages downstream as a mysterious accuracy drop.

- **📚 LEARNING TOPIC — "Which 'distances' are not metrics, and where that
  bites."** DTW, KL divergence, cosine distance, squared-Euclidean — each breaks
  a different metric axiom, and each breaks a different downstream assumption
  (k-NN graphs, ρ/σ bandwidths, embedding via MDS). A table of "axiom violated →
  algorithm that silently misbehaves" is a genuinely useful reference page, and
  `kernel.py` already exposes a `metric=` hook (only `"euclidean"` wired) that's
  the exact place this matters.

## MM9 — the sparsifier rounds *up*, and that's what makes it idempotent

`sparsify` thresholds |a| at the `sparsity`-quantile and binarises. Two
properties, both with a one-line "why":

1. **Achieved ≥ requested.** It uses `np.percentile(..., interpolation="higher")`
   and zeros everything `≤ thresh`. The "higher" rule rounds the threshold *up*
   to an actual data value, so on ties the realised zero-fraction never
   *undershoots* the target — the docstring's "at least the requested sparsity"
   is a direct consequence of that one kwarg.
2. **Idempotence.** Re-sparsifying the binary output at the same target is a
   no-op. Why: the first pass leaves a {0,1} matrix whose zero-fraction f ≥ s;
   the second pass's s-quantile then lands *inside* the zero block, so the
   threshold is 0, and "zero everything ≤ 0" fixes the zeros while leaving the
   ones. A fixed point. (The knife-edge f = s exactly, where "higher" would jump
   the threshold to 1 and wipe everything, is avoided precisely because pass one
   over-zeros — f > s strictly on continuous data.)

- **📚 LEARNING TOPIC — "Quantile interpolation conventions are not cosmetic."**
  `higher`/`lower`/`nearest`/`linear` change which side of a tie you land on, and
  here that choice is the difference between idempotent and not, and between
  "≥ target" and "≈ target." A small page on how the percentile convention
  propagates into a downstream invariant. (Tangent caught in passing: the source
  still uses the deprecated `interpolation=` kwarg, renamed to `method=` in numpy
  1.22 — noted, not touched, per the minimal-change policy.)

## MM11 — proving the legacy gets to retire

`graph/unionfind.py` is a hand-rolled disjoint-set predating
`scipy.cluster.hierarchy.DisjointSet`; its own docstring says it could be swapped
for scipy "once an equivalence test (MM11) lands." MM11 is that test: random
merge sequences must yield an *identical connected-components partition* under
both, and `solve_union_find` must return each group's full transitive closure
(cross-checked against scipy's `.subset`). The interesting bit is the framing —
the test isn't "does union-find work," it's "are these two implementations
*substitutable*." That's a **characterisation / differential test against a
trusted reference**, and it's what licenses a future refactor to delete code.

- **📚 LEARNING TOPIC — "Union-find as the canonical near-linear algorithm."**
  Path compression + union by rank gives the inverse-Ackermann α(n) amortised
  bound — the textbook example of an algorithm that's *effectively* O(1) per op
  but provably not. The hand-rolled version here uses *path splitting* (a
  one-pass variant of compression); worth a page on the family (compression /
  halving / splitting) and why they're all α(n). Connects to where SHREC uses it:
  merging recurrence-equivalent timepoints in the classical (Sauer) baseline.
- **📚 LEARNING TOPIC — "Differential testing to license deletion."** The general
  pattern: to retire bespoke code, don't just test it in isolation — pin it
  *equal to* the replacement on random inputs. Green test ⇒ the swap is
  behaviour-preserving. Pairs with the MM12 vectorised-vs-loop parity test
  (same idea, applied to `common_neighbors_ratio`).

## The real find: brentq controls the *root*, not the *residual*

Adding MM7/8/9/11 was meant to be routine, but rerunning the suite, the existing
Hypothesis property test (`test_sigma_solve_satisfies_equation_or_takes_fallback`)
fell over on a fresh adversarial input: a row with two neighbours separated by
ε ≈ 2×10⁻⁹. Diagnosis, because it's a clean lesson in numerical analysis:

- The σ-defining equation `f(σ) = Σ_m exp(−ReLU(d_m − ρ)/σ) − log₂k` is solved by
  `brentq`, which guarantees the **root location** to a tolerance: `|σ − σ*| < xtol`.
- A near-tie forces a *tiny* root, σ ~ ε. There, `f` is extremely steep:
  `f'(σ) ~ (ε/σ²)·e^{−ε/σ} ~ 1/σ`.
- The **function residual** at the returned root is `|f(σ)| ≈ f'(σ*)·|σ − σ*| ~
  xtol/σ ~ xtol/ε`. As the tie tightens (ε → 0), the residual *grows* even though
  the root is found perfectly. With `xtol = 1e-12` and ε ~ 2e-9 the residual was
  ~1.6e-6 — just over the test's optimistic 1e-6 bound.

So the test was making a subtly wrong claim: it conflated "root located
accurately" with "residual small." Those differ by the local slope. The fix
*names the regime* rather than loosening a magic number: accept a small residual,
**or** the exact tied fallback (σ = ρ), **or** a genuine steep root (σ ≪
neighbourhood scale). Crucially that third branch is **scale-invariant** (σ and
the scale both scale with the data, per MM2), so it can't be scaled away — and it
stays disjoint from the failure mode the test exists to catch: the old `fsolve`
stall parked σ at ρ (σ/scale ~ O(1)) with residual ≈ 2.8.

- **📚 LEARNING TOPIC — "Root tolerance vs residual tolerance."** The single most
  transferable nugget in this round. A bracketing solver promises `|x − x*|`
  small; it says *nothing* directly about `|f(x)|`, which is `≈ |f'(x*)|·|x − x*|`.
  For steep functions the two diverge by orders of magnitude. The remedies —
  tighten `xtol`, solve a rescaled/“flattened” equation, or assert on the root
  not the residual — are a core piece of practical numerics, and this repo has a
  live example. (Mirror image of the MM1 lesson, where `fsolve`'s *step-size*
  convergence test stalled with a tiny residual *claim* but a huge actual
  residual. Same theme from both ends: know exactly what your solver's
  convergence criterion measures.)
- **📚 LEARNING TOPIC — "Hypothesis as an adversary that never sleeps."** This
  edge wasn't in any hand-written oracle; property-based search *manufactured* a
  pathological near-tie. Lesson: a green property test isn't "proven correct," it's
  "not-yet-falsified," and rerunning it is cheap insurance that occasionally pays
  out a real refinement — here, a more precise statement of what the solver
  guarantees. Worth contrasting with example-based tests (which only ever check
  what you thought of) on the pedagogy page.

---

# Round 8 — closing the pipeline invariances: MM15, MM16, MM17 (2026-06-01)

Second follow-up batch, and it closes §5b.2 entirely. Where §5b.1 probes single
operators, §5b.2 asserts properties of the *whole* `fit_predict` that follow from
symmetry arguments, not from algorithm internals. These are the tests that say
"the method respects the structure of the problem" — and each one corresponds to
a real degree of freedom in the data that *should not* matter.

## MM15 — standardisation buys you the full similarity group

MM2 proved the affinity is invariant to a *global* rescale of all distances.
MM15 is the pipeline-level upgrade: scale and shift *each response independently*
(`x_k → a_k x_k + b_k`) and the recovered driver is unchanged. The mechanism is
the default `standardize=True`: a per-column z-score (`StandardScaler` +
`standardize_ts`) strips each channel's scale `a_k` and offset `b_k` *before*
embedding. So the invariance is the composition: standardisation kills the
per-channel affine, then MM2's scale-invariance handles whatever global scale is
left. The test deliberately uses *different* `a_k` per channel — a single global
factor would be a weaker claim already covered by MM2.

- **📚 LEARNING TOPIC — "Invariance by preprocessing vs invariance by
  construction."** Two ways a method can ignore a nuisance transform: the
  *operator* is intrinsically blind to it (MM2/MM7 — geometry), or a
  *preprocessing step* removes it first (MM15 — standardisation). The distinction
  matters because the second kind is only as trustworthy as the preprocessing:
  turn `standardize` off and MM15 should *fail*, while MM7 holds regardless. A
  good page would show the same invariance test passing/failing as you toggle the
  responsible flag — it makes "where does this invariance come from?" concrete.
- **📚 LEARNING TOPIC — "Why driver reconstruction must be affine-blind."** The
  physical story: each response measures the hidden driver through its own
  unknown gain and offset (sensor calibration). A method that wasn't affine-blind
  would conflate "two sensors with different gains" with "two different drivers."
  This is the measurement-model justification for standardisation, and ties back
  to the linear-measurement oracle MM25 still on the deferred list.

## MM16 — "more of the same data" is not more information

Consensus aggregation is a plain mean over responses. So stacking k identical
copies of the ensemble changes nothing: `(k·ΣAᵢ)/(k·N) = (ΣAᵢ)/N`. The test
pins `fit([X, X]) == fit(X)`. It reads trivial, but it's a sharp guard on one
specific bug: if the consensus step ever *summed* affinities without dividing by
the response count (or divided by a fixed constant), duplication would change the
graph's edge weights and could shift the spectral split or the Leiden resolution.
Idempotence-under-duplication is the cleanest statement that aggregation is a
genuine *average*, not an accumulation.

Implementation footnote worth keeping: I first tried the textbook form — one
response vs N copies of that *single* response — and hit a real quirk:
`standardize_ts` squeezes an `(T, 1)` array to 1-D, which `_make_embedding` then
rejects. Rather than "fix" a research-code path that N=1 was never meant to take,
the property is stated on the *ensemble* (duplicate all columns). Same math,
avoids the degenerate shape. (Lesson: when a test trips over a genuine edge in
the code, first ask whether the *property* can be stated without entering that
edge — often the cleaner statement is also the more general one.)

- **📚 LEARNING TOPIC — "Mean vs sum: the renormalisation that makes consensus
  scale-free in N."** Because aggregation is a mean, the consensus graph's edge
  weights live in [0,1] regardless of how many responses you have — which is what
  lets the *same* Leiden resolution / Fiedler threshold work from N=2 to N=200.
  An accumulating sum would make every tunable implicitly N-dependent. Connects
  directly to the MM27 accuracy-scaling story (adding responses sharpens the
  consensus, but doesn't rescale it).

## MM17 — refuse the degenerate input, loudly

A constant response has an all-zero distance matrix, so the kernel's
`surprise = dmat / np.std(dmat)` is `0/0`. `data_to_connectivity` guards this:
it detects channels equal to their own first timepoint, warns, and drops them.
MM17 pins both halves — (a) the warning fires, (b) the result is *exactly* the
result of the surviving channels — plus the negative: no warning when every
channel varies. Testing the *negative* is the part people skip; without it, a bug
that warns spuriously on good data would pass unnoticed.

- **📚 LEARNING TOPIC — "Defensive guards deserve two-sided tests."** Any
  `if degenerate: warn-and-skip` branch has two failure modes: missing a real
  degenerate (false negative) and firing on healthy input (false positive). A
  guard test that only feeds the bad case checks one. The `warnings.simplefilter
  ("error")` trick — promote any warning to an exception on the clean input — is
  the cheap way to pin the other side.
- **📚 LEARNING TOPIC — "0/0 in similarity kernels and where else it hides."**
  The constant-series NaN is one instance of a general hazard: any normalisation
  by a spread (std, range, max-distance) blows up on a degenerate input
  (constant, single-point, all-tied). The fuzzy-simplicial σ-solve had its own
  version (the fully-tied neighbourhood fallback in Round 1). A page cataloguing
  "normalise-by-spread" sites across the pipeline and how each handles the
  zero-spread limit would be a strong robustness reference — and three of the MM
  tests (MM6 underflow mask, the σ tied-fallback, MM17) are already worked
  examples of it.

## Where the catalog stands after two batches

§5b.1 and §5b.2 are closed (27 green, 2 documented xfail, 7 deferred). What's
left is all in §5b.3/§5b.4: the *driver-recovery* oracles (does the method
actually reconstruct a known `z(t)`? — MM24 identity, MM25 linear, MM26 monotone,
MM21 stochastic-forcing, MM23 cycle-graph Fiedler) and two slow paper-claim
regressions (MM28 percolation, MM29 HN baseline). Those are a qualitatively
different kind of test — end-to-end "does the science work," not "is the operator
well-behaved" — so they're the natural next theme to write up.

---

# Round 9 — driver recovery and the science oracles: MM23, MM24-26, MM21 (2026-06-01)

Third follow-up batch. This is where the catalog stops asking "is the operator
correct?" and starts asking "does the method *recover the hidden driver*?" — the
actual scientific claim of the paper. Four of these landed clean; the fifth
(MM21) landed as a measured *negative* that's arguably the most informative of
the lot.

## MM23 — a degenerate eigenspace, and why "Fiedler = cos" is wrong-by-a-phase

The cycle graph C_n has a *circulant* Laplacian, so its eigenvectors are the
Fourier modes and its eigenvalues are `λ_m = 2 − 2cos(2π m/n)`. The catalog
specced MM23 as "Fiedler is `cos(2π i/n)` up to sign." Writing it forced the
correction: `λ_1 = λ_{n−1}` are **equal** — the Fiedler eigenvalue is *doubly
degenerate*. Its eigenspace is the 2-D plane span{`cos(2π i/n)`, `sin(2π i/n)`},
and `eigh` is free to return *any* orthonormal basis of that plane. So the
Fiedler vector is fixed only up to **sign and phase**, not sign alone — asserting
`|cos(pred, cosine-mode)| ≈ 1` would be flaky, failing whenever the solver
happened to return something nearer the sine mode.

The fix is to assert the *subspace*, not the vector: project the recovered
eigenvector onto the first-harmonic plane and demand it keeps >99.9% of its norm
(and <5% leaks into the *second* harmonic, proving it's the lowest mode, not an
overtone). This is the right shape of claim whenever an eigenvalue is degenerate.

- **📚 LEARNING TOPIC — "Degenerate eigenspaces: test the subspace, not the
  vector."** Whenever a symmetry forces repeated eigenvalues (cyclic → Fourier
  pairs, spherical → spherical harmonics, any graph automorphism group), the
  individual eigenvectors are basis-arbitrary and only the eigen*space* is
  well-defined. The robust numerical test is a projector / principal-angle check.
  A genuinely reusable lesson — the single most common cause of "my eigenvector
  test is flaky" is testing a vector inside a degenerate space.
- **📚 LEARNING TOPIC — "Circulant matrices diagonalise in the Fourier basis."**
  Ring graphs, convolution operators, periodic boundary conditions — all
  circulant, all share the DFT eigenbasis with eigenvalues = DFT of the first
  row. A short derivation (why `λ_m = 2 − 2cos(2π m/n)` for C_n, and why the
  small-m modes are the smooth/low-frequency ones the Fiedler vector picks) ties
  spectral graph theory to signal processing in one picture.

## MM24-26 — measurement-function invariance, and the phase-vs-value trap

The three recovery oracles share one idea: SHREC should reconstruct the driver
`z(t)` up to a **monotone transform**, no matter how each sensor distorts it —
identity (MM24), affine (MM25), or saturating `tanh` (MM26). Spearman |ρ| is the
right yardstick precisely because it is itself monotone-invariant: it measures
"same ordering," which is exactly the indeterminacy the method carries. All three
recover the driver at |ρ| ≈ 0.997 — essentially perfectly — because monotone
observations of one driver have delay-embeddings that are all reparametrisations
of the same curve, so the consensus recurrence collapses to a single 1-D
coordinate tracking the driver's value.

The non-obvious part — and a genuine design decision — is the **driver had to be
aperiodic**. My first instinct was a sine driver (like MM27 uses). But a periodic
driver revisits every value twice per cycle, so the recurrence manifold recovers
*phase* (a bijection with time within a cycle), not *value* — and Spearman-vs-
value would be misleadingly low even though recovery "worked." MM27 gets away
with a sine because its responses are *chaotic logistic maps* whose recurrence is
governed by the forcing *value* at each step, so co-recurrence means equal `z`
regardless of phase. Here the responses are *direct* observations, so a smooth
*aperiodic* driver (Gaussian-smoothed noise) is needed to make value and
manifold-coordinate monotone-related.

- **📚 LEARNING TOPIC — "Phase vs value: what a 1-D manifold of a time series
  actually recovers."** The crux of delay-embedding intuition. For a periodic
  signal the recurrence manifold parametrises the *limit cycle* (phase); for a
  driver-forced ensemble it can parametrise the *forcing value*. Which one you get
  depends on whether co-recurrence is dominated by phase-coincidence or value-
  coincidence — and that depends on the measurement model. This single
  distinction explains why MM24-26 need an aperiodic driver while MM27 doesn't,
  and it's probably the deepest single idea in this whole notes file. Worth a
  figure: same driver, periodic vs aperiodic, manifold-coordinate vs z.
- **📚 LEARNING TOPIC — "Spearman as the natural metric for an up-to-monotone
  reconstruction."** Why correlation choice encodes your invariance: Pearson
  demands linearity, Spearman demands only monotonicity, mutual information
  demands only dependence. SHREC's output is defined up to a monotone transform,
  so Spearman is not a convenience — it's the metric that matches the method's
  own ambiguity. Pairs with the MM14 cosine-vs-Pearson discussion (choosing the
  metric to match the invariance).

## MM21 — the catalog was optimistic, and measuring it is the result

The catalog wanted period-8 at ARI > 0.85. It doesn't happen: measured ARI ≈
0.64 (discrete) and |ρ| ≈ 0.44 (continuous). This is the *same* representational
limit MM20 found for period-4, and it gets *worse* as the number of driver levels
grows — exactly what the "binary recurrence can't separate adjacent levels"
story (Round 6) predicts. So MM21, like MM20's companion, lands as a
**characterisation of a limitation**, not a green checkmark on a capability:
pin recovery into a partial-collapse band (`0.25 < ARI < 0.85`, `|ρ| < 0.85`) —
below target so the limit is real, above chance so structure is partially there —
and let it fire if a future representation ever breaks past 0.85.

- **📚 LEARNING TOPIC — "When the spec is wrong, the measurement is the
  deliverable."** The catalog's 0.85 was a guess; the honest move isn't to tune
  until something passes, it's to *measure* and pin the truth — even when the
  truth is "this doesn't reach the target, and here's the band it actually lives
  in." Two MM ids (MM20, MM21) now encode the same representational limit at two
  resolutions, which is far more informative than two optimistic green ticks would
  have been. (Contrast with the MM27 lesson: there the law held and we asserted
  its content; here the claim failed and we asserted *that*, precisely.)
- **📚 LEARNING TOPIC — "Resolution scaling of an information bottleneck."** MM19
  (2 levels, ARI≈0.99) → MM20 (4 levels, ARI≈0.50) → MM21 (8 levels, ARI≈0.64
  discrete / 0.44 continuous). The trend (not the individual numbers) is the
  physics: how many distinguishable driver states the consensus recurrence graph
  can encode before adjacent levels merge. A plot of recovery vs level-count would
  make the bottleneck quantitative and is the natural sequel to the Round 6
  binary-recurrence discussion.

## Catalog status after three batches

27 → **32 green**, 2 xfail, **2 deferred**. Sections §5b.1, §5b.2, §5b.3 are all
closed. The only open work is §5b.4's two slow paper-claim regressions: MM28
(percolation order parameter `T_LCC/T` vs N, Appendix E.3) and MM29 (Hirata-Nomura
Isomap baseline sanity).

---

# Round 10 — the baseline and a percolation dead-end worth recording (2026-06-01)

Fourth batch, the §5b.4 paper-claim regressions. One landed; one is deliberately
*not* forced — and the reason it isn't is the lesson.

## MM29 — pin the contract at the boundary, even when an inner test exists

`HirataNomuraIsomap` is the comparison baseline. Its consensus similarity feeds
`Isomap(metric='precomputed')`, which silently *requires* a symmetric,
zero-diagonal, non-negative dissimilarity. MM12 already proves
`common_neighbors_ratio` has those properties element-wise — so is MM29
redundant? No: MM12 tests the *function*; MM29 tests the *property at the place
Isomap depends on it*. If a future change routed a different matrix into Isomap,
or post-processed `common_neighbors_ratio`, MM12 would still pass while the
baseline silently fed Isomap a non-metric. Pinning the contract at the consumer
boundary is a distinct guard from pinning the producer.

- **📚 LEARNING TOPIC — "Producer tests vs consumer-contract tests."** The same
  property (here: valid dissimilarity) deserves a test where it's *produced* and
  another where it's *consumed*, because refactors break the link between them.
  A small but real principle for layered pipelines.

## MM28 — when a faithful reproduction needs the *right* construction, not a passing number

The percolation claim (Appendix E.3): as you add responses, the consensus
recurrence graph fragments, so the largest-connected-component fraction `T_LCC/T`
falls monotonically with N. Easy to state; I could not make it hold on the
*simplicial consensus mean* with a fixed-quantile threshold, and the failure is
instructive:

- At a median threshold, `T_LCC/T` went `[1.0, 1.0, 1.0, 0.5]` for one seed
  (a clean percolation transition into the two period-2 basins) — but another
  seed stayed at `1.0` throughout, and a third *rose*. Right at criticality the
  order parameter is wildly seed-dependent.
- Worse, seed-averaging didn't rescue monotonicity: at large N the mean `T_LCC/T`
  went back *up*. The reason is structural — `data_to_connectivity2` is a *mean*
  over channels, so as N grows the weight distribution **concentrates** (more
  channels averaged → smoother, more uniform weights), and a threshold defined
  *relative* to that distribution (a quantile) re-admits edges and the graph
  *re-percolates*. The order parameter is measuring the threshold's drift, not
  the physics.

The honest conclusion: a faithful Appendix-E.3 reproduction almost certainly
uses the **binary Sauer recurrence graph** (`ClassicalRecurrenceClustering` /
`kernel.py`, the `inf_k d^(k)` co-occurrence graph) at an **absolute** distance
threshold — where "add a response" means "intersect one more recurrence
constraint," which can only *remove* edges, giving a genuinely monotone
fragmentation. So MM28 stays deferred with that finding recorded, rather than
shipped with a quantile hand-tuned until one configuration happened to drop 0.3.

- **📚 LEARNING TOPIC — "Relative vs absolute thresholds under a shifting
  distribution."** A quantile threshold tracks the data distribution; an absolute
  threshold is fixed. When the distribution itself moves with the independent
  variable (here, N), a relative threshold confounds the effect you're trying to
  measure. The percolation order parameter only means what the paper intends if
  "more data" can only *remove* edges — which an intersection of binary
  constraints guarantees and a renormalising mean does not. A genuinely subtle
  measurement-design point, and the cleanest example of it in the whole project.
- **📚 LEARNING TOPIC — "Monotone-by-construction vs monotone-by-luck."** The
  binary-intersection recurrence graph is monotone in N *by construction* (edges
  only disappear); the simplicial-mean graph is not. Recognising which quantities
  are monotone for structural reasons — versus which merely trend that way on
  average — is what separates a robust regression test from a flaky one. This is
  why MM28 is worth doing *right* rather than *now*.
- **Process lesson (mirrors Round 6's MM20):** a clean *negative* — "this
  construction can't show the claimed effect, and here's the mechanism" — is a
  real deliverable. It saves the next person the same dead-end and points them at
  the construction that will work. Recording it in the catalog (status
  "calibration blocked," with the mechanism) is more honest than either a forced
  green or a bare "deferred."

## Catalog status after four batches

**33 green, 2 xfail, 1 deferred.** §5b.1/§5b.2/§5b.3 fully closed; §5b.4 has MM27
+ MM29 green and only MM28 open (calibration-blocked, mechanism recorded above).
From the session's starting point (the math-correctness suite as inherited), the
post-PR follow-up added MM7-9, MM11, MM15-17, MM21, MM23-26, MM29 and refined the
σ-solver property test — every deferred "must" and non-must oracle is now either
green, a documented xfail, or a deferred-with-mechanism. The pedagogy file has
ten rounds of `📚 LEARNING TOPIC` seeds ready for the Quarto explainer.
