"""Continuous-driver SHREC model — Fiedler eigenvector of the consensus
recurrence Laplacian (paper Appendix B step 5).

A preset over `ShrecPipeline`: `SimplicialConnectivity` + `FiedlerReconstructor`.
"""
from shrec.models.pipeline import ShrecPipeline
from shrec.recurrence.connectivity import SimplicialConnectivity
from shrec.reconstruct import FiedlerReconstructor


class RecurrenceManifold(ShrecPipeline):
    """Reconstruct a continuous driver signal by spectral embedding of the
    consensus recurrence graph.

    Parameters
    ----------
    n_components : int
        Number of non-trivial Laplacian eigenvectors to return (the driver
        embedding dimension). The Fiedler vector is `n_components=1`.
    normalize_laplacian : bool
        If False (default), use the unnormalised Laplacian `L = D − A`
        (RatioCut), matching the paper. If True, use the NCut / random-walk
        normalisation (generalised eigenproblem `L v = λ D v`), which corrects
        for degree heterogeneity across responses ("response bias").
    k, tol, aggregation
        Simplicial-connectivity knobs (paper Appendix B step 3/4); the
        distance `metric` is inherited from the base model. Defaults
        reproduce the canonical pipeline exactly.
    """

    def __init__(
        self,
        n_components=1,
        normalize_laplacian=False,
        k=10,
        tol=1e-5,
        aggregation="mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.normalize_laplacian = normalize_laplacian
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
        return FiedlerReconstructor(
            n_components=self.n_components,
            normalize_laplacian=self.normalize_laplacian,
            verbose=self.verbose,
        )
