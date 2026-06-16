"""Classical Sauer-style recurrence clustering baseline (PRL 2004).

A preset over `ShrecPipeline`: `ExpKernelConnectivity` (sparsified) +
`UnionFindReconstructor`.
"""
from shrec.models.pipeline import ShrecPipeline
from shrec.recurrence.connectivity import ExpKernelConnectivity
from shrec.reconstruct import UnionFindReconstructor


class ClassicalRecurrenceClustering(ShrecPipeline):
    """Cluster a time series using Sauer (PRL 2004) union-find equivalence
    classes on a binarised recurrence matrix.

    This is the noise-free Sauer-limit baseline. For the canonical SHREC
    pipeline (adaptive simplicial complex + Leiden), use
    `RecurrenceClustering` / `RecurrenceManifold` instead.

    Parameters
    ----------
    scale : float
        Distance rescaling for the exp-of-distance kernel.
    order : float
        p-norm ensemble aggregation exponent; large values approach the
        Sauer `inf_k` (min-over-channels). The sparsify target and the
        clique weighting come from the base `tolerance` /
        `weighted_connectivity`.
    """

    def __init__(self, scale=1.0, order=500.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.order = order

    def _connectivity_stage(self):
        return ExpKernelConnectivity(
            scale=self.scale, order=self.order, time_exclude=0,
            metric=self.metric, sparsify=True, tolerance=self.tolerance,
            weighted=self.weighted_connectivity,
        )

    def _reconstructor_stage(self):
        return UnionFindReconstructor()
