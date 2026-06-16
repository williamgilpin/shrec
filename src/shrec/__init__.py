"""shrec — Shared Recurrence (Gilpin, Phys. Rev. X 15, 011005, 2025).

Top-level public API. Concrete models can be imported directly as
``from shrec import RecurrenceManifold``; sub-packages remain available
for advanced use (`shrec.recurrence`, `shrec.graph`, `shrec.utils`).
"""
__version__ = "0.2.0.dev0"

from shrec import graph, models, recurrence, reconstruct, utils
from shrec.embeddings import embed_ts, hankel_matrix, make_embedding
from shrec.models import (
    ClassicalRecurrenceClustering,
    HirataNomuraIsomap,
    RecurrenceClustering,
    RecurrenceManifold,
    ShrecPipeline,
)

__all__ = [
    "__version__",
    # composable pipeline + the four named presets
    "ShrecPipeline",
    "RecurrenceClustering",
    "RecurrenceManifold",
    "ClassicalRecurrenceClustering",
    "HirataNomuraIsomap",
    # embedding helpers
    "embed_ts",
    "hankel_matrix",
    "make_embedding",
    # sub-packages
    "graph",
    "models",
    "recurrence",
    "reconstruct",
    "utils",
]
