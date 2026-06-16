"""Discrete-driver SHREC model — Leiden community detection on the
consensus fuzzy simplicial complex (paper Appendix B steps 3–5).

A preset over `ShrecPipeline`: `SimplicialConnectivity` + `LeidenReconstructor`.
"""
from shrec.models.pipeline import ShrecPipeline
from shrec.recurrence.connectivity import SimplicialConnectivity
from shrec.reconstruct import LeidenReconstructor


class RecurrenceClustering(ShrecPipeline):
    """Assign a discrete cluster label to each timepoint based on community
    structure in the consensus recurrence graph. Best suited to discrete-
    time driver signals.

    Parameters
    ----------
    resolution, objective, method
        Leiden community-detection knobs (`graph.communities._leiden`).
    k, tol, aggregation
        Simplicial-connectivity knobs (paper Appendix B step 3/4); the
        distance `metric` is inherited from the base model. Defaults
        reproduce the canonical pipeline exactly.
    """

    def __init__(
        self,
        resolution=1.0,
        objective="modularity",
        method="graspologic",
        k=10,
        tol=1e-5,
        aggregation="mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resolution = resolution
        self.objective = objective
        self.method = method
        self.k = k
        self.tol = tol
        self.aggregation = aggregation

    def _connectivity_stage(self):
        return SimplicialConnectivity(
            k=self.k, tol=self.tol, time_exclude=self.time_exclude,
            aggregation=self.aggregation, metric=self.metric,
            verbose=self.verbose,
        )

    def _reconstructor_stage(self):
        return LeidenReconstructor(
            resolution=self.resolution, objective=self.objective,
            method=self.method, random_state=self.random_state,
        )
