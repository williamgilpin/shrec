"""Hirata–Nomura recurrence-manifold baseline.

Recurrence-manifold reconstruction (Hirata et al. 2008) using the
common-neighbour-ratio consensus similarity matrix of Nomura et al.
(2022), embedded with Isomap.

A preset over `ShrecPipeline`: `CommonNeighborsConnectivity` +
`IsomapReconstructor`.
"""
from shrec.models.pipeline import ShrecPipeline
from shrec.recurrence.connectivity import CommonNeighborsConnectivity
from shrec.reconstruct import IsomapReconstructor
from shrec.utils import common_neighbors_ratio


class HirataNomuraIsomap(ShrecPipeline):
    """HN-Isomap baseline for continuous driver reconstruction."""

    def __init__(self, n_components=2, percentile=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.percentile = percentile

    def _connectivity_stage(self):
        return CommonNeighborsConnectivity(
            percentile=self.percentile, metric=self.metric,
        )

    def _reconstructor_stage(self):
        return IsomapReconstructor(n_components=self.n_components)

    def transform(self, X):
        X = self._preprocess(X)
        X = self._make_embedding(X)
        return common_neighbors_ratio(X)
