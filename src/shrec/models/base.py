"""Base estimator class shared by all SHREC models."""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import PowerTransformer, StandardScaler

from shrec.embeddings import embed_ts
from shrec.utils import (
    adjmat_from_associations,
    detrend_ts,
    hollow_matrix,
    standardize_ts,
)


class RecurrenceModel(BaseEstimator, ClusterMixin):
    """Base class for recurrent time-series models.

    Holds the shared preprocessing / embedding / hyperparameter machinery.
    Concrete model classes (`RecurrenceClustering`, `RecurrenceManifold`,
    `ClassicalRecurrenceClustering`, `HirataNomuraIsomap`) override `fit`.

    Attributes
    ----------
    tolerance : float
        Expected fraction of recurrence events in the time series. Increasing
        this improves noise robustness at the cost of driver resolution.
    d_embed : int
        Embedding dimension (number of past timepoints to stack).
    noise : float
        Std-dev of optional i.i.d. Gaussian noise added to the embedded data
        as a regulariser.
    random_state : int or None
        Random seed used by stochastic backends (Leiden, embedding noise).
    standardize, power_transform, detrend : bool
        Optional preprocessing toggles.
    weighted_connectivity : bool
        Whether `_neighbors_to_cliques` produces a weighted output.
    time_exclude : int
        Number of neighbour timepoints around each row to mask out.
    metric : str
        Distance metric forwarded to `scipy.spatial.distance.cdist` by the
        connectivity stage (any cdist-supported metric).
    padding : "symmetric" | "constant" | None
        `numpy.pad` mode for the delay embedding.
    fill_nan : bool
        Forward/back-fill NaN values in the input before embedding.
    verbose : bool
        Print progress updates from long-running steps.
    """

    def __init__(
        self,
        tolerance=0.01,
        d_embed=3,
        noise=0.0,
        random_state=None,
        make_embedding=True,
        time_exclude=0,
        standardize=True,
        power_transform=False,
        weighted_connectivity=True,
        use_sparse=False,
        store_adjacency_matrix=False,
        detrend=False,
        metric="euclidean",
        padding="symmetric",
        fill_nan=False,
        verbose=False,
    ):
        self.tolerance = tolerance
        self.make_embedding = make_embedding
        self.d_embed = d_embed
        self.noise = noise
        self.random_state = random_state
        self.time_exclude = time_exclude
        self.weighted_connectivity = weighted_connectivity
        self.standardize = standardize
        self.power_transform = power_transform
        self.use_sparse = use_sparse
        self.store_adjacency_matrix = store_adjacency_matrix
        self.padding = padding
        self.detrend = detrend
        self.metric = metric
        self.fill_nan = fill_nan
        self.verbose = verbose

    def _fillna(self, X):
        """Forward-fill then back-fill NaN entries in an input array."""
        Xc = pd.DataFrame(X)
        Xc = Xc.ffill()
        Xc = Xc.bfill()
        return Xc.values

    def _make_embedding(self, X):
        """Time-delay embed the input; broadcast over the channel axis."""
        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0], -1))
            X_embed = embed_ts(X, self.d_embed, padding=self.padding)
        elif len(X.shape) == 3:
            warnings.warn(
                "Multivariate time series detected, embedding each "
                "dimension separately."
            )
            all_embeddings = []
            for i in range(X.shape[-1]):
                slice_2d = np.reshape(X[..., i], (X.shape[0], -1))
                all_embeddings.append(
                    embed_ts(slice_2d, self.d_embed, padding=self.padding)
                )
            X_embed = np.hstack(all_embeddings)
        else:
            raise ValueError("Input shape not valid.")

        if self.noise > 0.0:
            # Use a fresh Generator so we never touch the global numpy RNG
            # (audit §3A item A6: construction used to call `np.random.seed`,
            # silently mutating process-wide state).
            rng = np.random.default_rng(self.random_state)
            X_embed = X_embed + self.noise * rng.standard_normal(X_embed.shape)

        return X_embed

    def _preprocess(self, X):
        """Apply the standard preprocessing chain (detrend → power → standardise → nan-fill)."""
        if self.detrend:
            X = detrend_ts(X)
        if self.power_transform:
            X = PowerTransformer().fit_transform(X)
        if self.standardize:
            X = StandardScaler().fit_transform(X)
            X = standardize_ts(X)
        if self.fill_nan:
            X = self._fillna(X)
        return X

    def _find_distance_matrix(self, X):
        """Dense per-channel pairwise distance stack. Currently unused — kept
        for benchmarks that depend on it; allocates an `(B, T, T, D)`
        temporary, so don't run on large inputs."""
        all_dist_mat = np.sqrt(
            np.sum((X[..., None, :] - X[:, None, ...]) ** 2, axis=-1)
        ).T
        all_dist_mat += (np.eye(all_dist_mat.shape[0]) * 1e16)[..., None]
        return all_dist_mat

    def _neighbors_to_cliques(self, bdmat):
        """Convert a binary neighbour matrix to a clique-augmented adjacency."""
        return adjmat_from_associations(
            hollow_matrix(bdmat),
            weighted=self.weighted_connectivity,
            use_sparse=self.use_sparse,
        )

    def fit_transform(self, X, y=None):
        """sklearn-style `fit_transform` — proxies to `fit_predict`."""
        return self.fit_predict(X, y)
