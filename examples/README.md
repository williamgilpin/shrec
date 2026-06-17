# Examples

Runnable, self-contained scripts (they synthesise their own data, so no data
files are needed). Each writes a PNG next to itself. They are the basis for the
`.ipynb` notebooks on the release punch list (`docs/pre-release-notes.md`).

| Script | What it shows |
|--------|---------------|
| `quickstart.py` | Response ensemble → recovered driver with `RecurrenceManifold`, plus the consensus graph. The minimal "does it work". |
| `pipeline_composition.py` | The stage-strategy API (decision 0008): a preset equals its explicit `ShrecPipeline(connectivity=…, reconstructor=…)`, and swapping the connectivity / aggregation / injecting a precomputed `A` are one-liners. |

Run:

```
uv run python examples/quickstart.py
uv run python examples/pipeline_composition.py
```

Plotting helpers live in `shrec.plotting` (requires the `shrec[viz]` extra:
`plot_driver_overlay`, `plot_recurrence_matrix`).
