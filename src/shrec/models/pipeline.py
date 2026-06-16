"""Composable SHREC pipeline — `Connectivity` × `Reconstructor`.

`ShrecPipeline` owns the invariant spine of every SHREC model (paper
Appendix B): preprocess → delay-embed → build the recurrence graph
(`Connectivity`) → reconstruct the driver (`Reconstructor`). The four named
models (`RecurrenceClustering`, `RecurrenceManifold`,
`ClassicalRecurrenceClustering`, `HirataNomuraIsomap`) are thin *presets*
that map their scalar knobs onto stage objects.

Power users compose stages directly::

    from shrec.models import ShrecPipeline
    from shrec.recurrence.connectivity import SimplicialConnectivity
    from shrec.reconstruct import FiedlerReconstructor, UnionFindReconstructor

    # simplicial recurrence + Sauer union-find (not a named preset)
    ShrecPipeline(
        connectivity=SimplicialConnectivity(k=15),
        reconstructor=UnionFindReconstructor(),
    ).fit(X)

See `docs/decisions/0008-pipeline-modularization.md`.
"""
import numpy as np

from shrec.models.base import RecurrenceModel
from shrec.recurrence.connectivity import SimplicialConnectivity
from shrec.reconstruct import FiedlerReconstructor


class ShrecPipeline(RecurrenceModel):
    """Compose a `Connectivity` stage and a `Reconstructor` stage into a
    sklearn-style estimator.

    Parameters
    ----------
    connectivity : Connectivity or None
        Embedded stack `(N, T, D)` → consensus affinity `(T, T)`. Defaults to
        the canonical `SimplicialConnectivity`.
    reconstructor : Reconstructor or None
        Affinity `(T, T)` → driver labels. Defaults to `FiedlerReconstructor`
        (continuous driver).
    """

    def __init__(self, connectivity=None, reconstructor=None, **kwargs):
        super().__init__(**kwargs)
        self.connectivity = connectivity
        self.reconstructor = reconstructor

    def _connectivity_stage(self):
        """The connectivity strategy for this fit. Presets override."""
        if self.connectivity is not None:
            return self.connectivity
        return SimplicialConnectivity(time_exclude=self.time_exclude)

    def _reconstructor_stage(self):
        """The reconstructor strategy for this fit. Presets override."""
        if self.reconstructor is not None:
            return self.reconstructor
        return FiedlerReconstructor()

    def fit(self, X, y=None):
        """
        Args:
            X (array-like): shape (n_timepoints, n_channels).
            y: ignored, present for sklearn API parity.
        """
        X = self._preprocess(X)
        X = self._make_embedding(X)

        affinity = self._connectivity_stage()(X)
        if self.store_adjacency_matrix:
            self.adjacency_matrix = affinity

        reconstructor = self._reconstructor_stage()
        labels = reconstructor(affinity)

        self.indices = np.arange(len(labels))
        self.labels_ = labels
        for key, val in reconstructor.extra_attrs(labels).items():
            setattr(self, key, val)
        return self
