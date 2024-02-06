import numpy as np
import scipy
import warnings
import datetime
try:
    import pandas as pd
    has_pandas = True
except ImportError:
    warnings.warn("Could not import pandas, missing value filling will not function.")
    has_pandas = False

from .utils import *
from .utils import embed_ts, standardize_ts, minmax_ts, find_psd, common_neighbors_ratio, nan_fill

# from sklearn.cluster import SpectralClustering
# from sklearn.manifold import SpectralEmbedding
# from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

def solve_union_find(lists):
    disjoint = DisjointSet()
    for lst in lists:
        disjoint.union(*lst)

    groups = disjoint.groups()
    return [lst and groups[disjoint.find(lst[0])] for lst in lists]


class DisjointSet:
    """
    A disjoint set data structure. 
    
    Adapted fro open-source code:
    https://stackoverflow.com/questions/67805907

    Attributes:
        elements (dict): A dictionary of elements in the set

    """

    class Element:
        def __init__(self):
            self.parent = self
            self.rank = 0

    def __init__(self):
        self.elements = {}

    def find(self, key):
        el = self.elements.get(key, None)
        if not el:
            el = self.Element()
            self.elements[key] = el
        else:  # Path splitting algorithm
            while el.parent != el:
                el, el.parent = el.parent, el.parent.parent
        return el

    def union(self, key=None, *otherkeys):
        if key is not None:
            root = self.find(key)
            for otherkey in otherkeys:
                el = self.find(otherkey)
                if el != root:
                    # Union by rank
                    if root.rank < el.rank:
                        root, el = el, root
                    el.parent = root
                    if root.rank == el.rank:
                        root.rank += 1

    def groups(self):
        result = {el: [] for el in self.elements.values() if el.parent == el}
        for key in self.elements:
            result[self.find(key)].append(key)
        return result


import scipy.sparse as sp
def _leiden(g, method="graspologic", objective="modularity", resolution=1.0, random_state=None):
    """
    Compute the Leiden clustering of a graph, represented by a numpy matrix or a sparse
    matrix.
    
    Args:
        g (ndarray or sparse_csr_matrix): a representation of a graph as a matrix or 
            sparse matrix.
        method ("graspologic" | "leidenalg" | "igraph" | "cdlib"): The algorithm to use
            to compute the Leiden clustering
        objective ("modularity" | "cpm") : the Leiden clustering method to use
        resolution (float): the resolution parameter to use for the Leiden clustering
            algorithm.
        random_state (int or None): the random state to use for the Leiden clustering
    
    Returns:
        indices (ndarray): An ordered list of indices of nodes in g
        labels (ndarray): A list of labels for each point in indices
    
    """
    if objective not in ["modularity", "cpm"]:
        warnings.warn("Objective function not recognized; falling back to modularity")
        objective = "modularity"

    ## Convert graph to appropriate format
    if method in ["igraph", "leidenalg"]:
        import igraph as ig

        g_ig = ig.Graph.Adjacency(g).as_undirected()

    if method in ["cdlib"]:
        import networkx as nx

        if sp.issparse(g):
            g_nx = nx.convert_matrix.from_scipy_sparse_matrix(g)
        else:
            g_nx = nx.convert_matrix.from_numpy_matrix(g)

    ## Find communities
    if method == "graspologic":
        import graspologic

        mod_flag = objective == "modularity"
        partition = graspologic.partition.leiden(
            g, use_modularity=mod_flag, resolution=resolution, random_seed=random_state,
        )
        indices, labels = np.array([(key, partition[key]) for key in partition]).T

    if method == "igraph":
        cluster_obj = g_ig.community_leiden(
            objective_function=objective, resolution_parameter=1.0
        )
        labels = cluster_obj.membership
        indices = np.arange(len(labels))

    if method == "cdlib":
        import cdlib

        if objective == "modularity":
            coms = cdlib.algorithms.leiden(g_nx)
        elif objective == "cpm":
            coms = cdlib.algorithms.cpm(g_nx)
        else:
            warnings.warn(
                "Objective function not recognized; falling back to modularity"
            )
        indices, labels = np.array(community_list_to_labels(coms.communities)).T

    if method == "leidenalg":
        import leidenalg as la

        if objective == "modularity":
            objective_obj = la.ModularityVertexPartition
        else:
            objective_obj = la.CPMVertexPartition
        cluster_membership = la.find_partition(g_ig, objective_obj)
        labels = cluster_membership._membership
        indices = np.arange(len(labels))

    ## sort output
    sort_inds = np.argsort(indices)
    indices, labels = indices[sort_inds], labels[sort_inds]

    return indices, labels


############################################################################
#
#
#              Distance matrix calculations
#
#
############################################################################



from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import scipy.sparse


from scipy.optimize import fsolve

relu = lambda x: np.maximum(0, x)
def dataset_to_simplex(X, k=20, tol=1e-6, precomputed=False):
    """
    Given a dataset X, return the fuzzy simplicial simplex formed by the points in X

    Args:
        X (np.ndarray): dataset of shape (n_samples, n_features)
        k (int): number of nearest neighbors to use
        tol (float): tolerance for the solver
        precomputed (bool): whether the input is a precomputed distance matrix. If False,
            the distance matrix will be computed from X along the first dimension

    Returns:
        wmat (np.ndarray): adjacency matrix of the simplex, with 
            shape (n_samples, n_samples)

    References:
        McInnes, Leland, John Healy, and James Melville. "Umap: Uniform manifold 
            approximation and projection for dimension reduction." 
            arXiv preprint arXiv:1802.03426 (2018).
    
    """
    if not precomputed:
        dmat = cdist(X, X)
    else:
        dmat = X.copy()

    n = dmat.shape[0]
    dmat_zerofilled = dmat.copy()
    dmat_zerofilled[dmat_zerofilled < 1e-10] = np.inf
    dists = np.partition(dmat_zerofilled, k + 1, axis=1) # distances to k nearest neighbors
    dists = np.sort(dists, axis=1)[:, 1:k+1]  # drop self
    # dists = np.sort(dmat, axis=1)[:, 1:k+1] # distances to k nearest neighbors
    rho = dists[:, 0] # distance to the nearest neighbor
    for i in range(n):
        # if dists[i, -1] < tol: # maximum distance is nearly zero
        #     dmat[i] = np.exp(-relu(dmat[i]) / (1e-8 + np.std(dmat[i])))
        #     continue
        func = lambda sig: sum(np.exp(-relu(dists[i] - rho[i]) / sig)) - np.log2(k)
        jac = lambda sig: sum(np.exp(-relu(dists[i] - rho[i]) / sig) * relu(dists[i] - rho[i])) / sig**2
        sigma_i = fsolve(func, rho[i], fprime=jac, xtol=tol)[0]
        dmat[i] = np.exp(-relu(dmat[i] - rho[i]) / sigma_i)
    wmat = dmat + dmat.T - dmat * dmat.T
    return wmat

from umap.umap_ import fuzzy_simplicial_set
from scipy.sparse import coo_matrix

def data_to_connectivity2(X, k=10, tol=1e-5, verbose=False):
    """
    Given a list of datasets, compute their simplicial complexes and average

    Args:
        X (np.ndarray): dataset of shape (n_samples, n_times, n_dims)
        k (int): number of nearest neighbors to use
        tol (float): tolerance for the solver
        verbose (bool): whether to print progress

    Returns:
        wmat (np.ndarray): adjacency matrix of the simplex, with 
            shape (n_samples, n_samples)
    
    """
    nb, nt, nd = X.shape

    if not verbose:
        warnings.filterwarnings('ignore') ## fsolve often throws warnings

    ## Use built-in implementation
    # wmat = np.zeros((nt, nt))
    # for ind, X0 in enumerate(X):
    #     if verbose and ind % (nb // 10) == 0:
    #         print(ind, "/", len(X), flush=True)
    #     wmat += dataset_to_simplex(X0, k=k, tol=tol) / nb

    wmat = coo_matrix(np.zeros((nt, nt)))
    for ind, X0 in enumerate(X):
        if verbose and ind % (nb // 10) == 0:
            print(ind, "/", len(X), flush=True)
        result, sigmas, rhos, dists = fuzzy_simplicial_set(X0, k, 0, 'euclidean', return_dists=True)
        # wmat += result / nb
        # wmat += np.array(result.todense()) / nb

        sigmas, rhos = np.asarray(sigmas), np.asarray(rhos)
        dmat = cdist(X0, X0)
        dmat = np.exp(-relu(dmat - rhos[None, :]) / sigmas[None, :])
        dmat = dmat + dmat.T - dmat * dmat.T # symmetrize
        wmat += dmat / nb

    wmat  = np.asarray(wmat)
    # wmat = np.array(wmat.todense())

    return wmat

def data_to_connectivity(X, 
                         time_exclude=0,  scale=1.0, 
                         ord=1.0, metric="euclidean"):
    """
    Given a stack of M time series, each of shape N x D, compute a 
    single consolidated N x N connectivity or adjacency matrix
    
    Args:
        X (array-like): A list of coordinates of shape M x N x D, where M is the number 
            of time series, N is the number of time points, and D is their 
            dimensionality
        metric ("euclidean" | "dtw"): The metric to use for distance computation
        time_exclude (float): The fraction of timepoints to exclude from neighbor 
            counts around each timepoint
        use_sparse (bool): Whether to use a sparse matrix representation of the distance
            matrix.
        scale (float): The scale of the distance matrix. If scale is 1, the distance
            matrix is unmodified.
        ord (float): The order of the aggregation norm. Defaults to a order 1, which
            is a simple mean.
        
    Returns:
        bd (array-like): A binarized distance matrix of shape N x N
    """
    
    sel_inds = np.all(np.isclose(X, X[:, :1, :], 1e-12), axis=(1, 2))
    if np.sum(sel_inds) > 0:
         warnings.warn(f"{np.sum(sel_inds)} Constant time series detected. " 
                        + "Skipping these datasets.")
    X = X[np.logical_not(sel_inds)]
    nb, nt, nd = X.shape
    
    thresh =  np.median(
        np.linalg.norm(X - np.median(X, axis=1, keepdims=True), axis=-1)
    )
    thresh = 1 / (scale * thresh)
    
    ## Iterate over all time series, compute distance matrix
    # dstack, bdstack = np.zeros((2, nb, nt, nt))
    bd = np.zeros((nt, nt))

    for i in range(nb):
        dmat = cdist(X[i], X[i]) # pairwise distance matrix
        ## Normalize and exponentiate to find connectivity matrix then mean merge
        surprise = dmat / np.std(dmat)
        #thresh = 1.0
        bd += (1 / nb) * np.exp(-surprise * ord / thresh) 
    
    # remove adjacent timescales by zeroing near-diagonals in distance matrix
    if time_exclude > 0:
        mask_mat = np.ones_like(bd)
        mask_mat = 1 - (
            np.triu(mask_mat, k=-time_exclude)
            * np.tril(mask_mat, k=time_exclude)
        )
        bd *= mask_mat.astype(float)

    # Compute p-norm
    bd = bd ** (1 / ord)

    return bd
    


from scipy.optimize import root_scalar

def distance_to_connectivity(dmat, dscale=None, sparsity=None):
    """
    Convert a distance matrix to a connectivity matrix via an exponential transform
    If no crossover scale is specified, one is automatically computed based on a target
    sparsity in the output matrix
    
    Args:
        dmat (array-like): An distance matrix of shape (N, N), or a stack of distance
            matrices of shape (B, N, N)
        dscale (float): The crossover scale for connectivity. This determines the 
            typical distance to be considered connected or not. This setting cannot
            be used with a target sparsity threshold
        sparsity (float): The target fraction of zero entries in the connectivity 
            matrix. This setting cannot be used with a target distance scale

    Returns:
        cmat (array-like): A connectivity matrix of shape (N, N), or a stack of matrices
    """
    if (dscale is None) and (sparsity is None):
        sparsity = 0.99  # default case

    if (dscale is not None) and (sparsity is not None):
        warnings.warn(
            "Both a distance scale and a sparsity have been specified; only the"
            + "distance scale will be used"
        )

    if dscale is not None:
        connectivity = np.exp(-dmat / dscale)
        return connectivity

    if sparsity is not None:
        dscale = -np.median(dmat) / np.log(1 - sparsity)
        denom = np.sum(np.ones_like(dmat))
        optfun = lambda x: np.sum(np.exp(-dmat / x)) / denom - (1 - sparsity)

        scale_factor = root_scalar(optfun, bracket=[1e-16, dscale]).root
        connectivity = np.exp(-dmat / scale_factor)
        return connectivity

    return connectivity





############################################################################
#
#
#              MODELS
#
#
############################################################################


from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
from scipy.optimize import minimize_scalar, root_scalar
from scipy.stats import boxcox
import warnings

class RecurrenceModel(BaseEstimator, ClusterMixin):
    """
    A base class for recurrent time series models. This class includes utilities for
    detecting recurrences, and consolidating distance matrices across an ensemble
    
    Attributes:
        tolerance (float): The expected fraction of recurrence events in the time series
            This defaults to 1% of all timepoints. Increasing this parameter improves 
            noise robustness, at the expense of decreasing the detail with which the
            driver can be resolved
        d_embed (int): The number of past timepoints to use for embedding
        noise (float): The amplitude of the noise used to blur the input data (for 
            regularization).
        eps (float): The tolerance for neighbor detection.
        random_state (int): The random seed for the random number generator
        make_embedding (bool): Perform a time delay embedding before computing the 
            distance matrix. If multivariate time series are passed, each channel is 
            treated as an independent time series. Otherwise, the time series are 
            embedded using the method of delays
        time_exclude (int): The number of neighboring timepoints to exclude from the 
            neighborhood calculation
        merge (str): the method of consolidating distances across different timepoints. 
            Defaults to the minimum distance ("min") observed across any response system. 
            Other options include the mean across systems ("mean") and the fifth 
            percentile ("percentile")
        standardize (bool): Whether to standardize the input time series
        power_transform (bool): Whether to apply a Power transformation to the input 
            time series featurewise, in order to gaussianize the features.
        detrend (bool): Whether to detrend the input time series
        use_sparse (bool): Whether to default to sparse matrices, allowing longer time 
            series at the expense of accuracy
        store_adjacency_matrix (bool): Whether to store the neighbor matrix
        padding ("symmetric" | "constant" | None): The method of padding the time delay 
            embedding. See the documentation for numpy.pad for all options.
        metric ("euclidean" or "dtw"): The metric to use to compute the
            pairwise distance matrix among timepoints
        scale (float): The scaling of the elements of the distance matrix before conversion
            to connectivity
        aggregation_order (float): The order of aggregation to use. The default of 1 
            corresponds to a simple average of the distance matrices, while
            a value near infinity corresponds to a maximum (L-\infty) aggregation.
        fill_nan (bool): Whether to fill NaN values in the input time series using 
            forward filling
        verbose (bool): Whether to print progress updates


    To do:
        Alternative distance metrics; implicit embedding via DTW calculation
        Matrix profile faster than computing full distance matrix
        Add a subclass for time series motif detection
        Can increase epsilon radius to recover information from a broader area
        Speed up distance matrix calculation.
        PCA performs worse when the driven dynamical systems are very different.
            Same is true for tuning curves?
        Improve sparse matrix calculations and KDTree call efficiency

    """

    def __init__(
        self,
        tolerance=0.01,
        d_embed=3,
        noise=0.0,
        eps=0.025,
        random_state=None,
        make_embedding=True,
        time_exclude=0,
        standardize=True,
        power_transform=False,
        weighted_connectivity=True,
        merge="min",
        use_sparse=False,
        store_adjacency_matrix=False,
        detrend=False,
        metric="euclidean",
        scale=1.0,
        aggregation_order=1.0,
        padding="symmetric",
        fill_nan=False,
        verbose=False,
    ):

        self.tolerance = tolerance
        self.eps = eps
        self.make_embedding = make_embedding
        self.d_embed = d_embed
        self.noise = noise
        self.random_state = random_state
        self.time_exclude = time_exclude
        self.merge = merge
        self.weighted_connectivity = weighted_connectivity
        self.standardize = standardize
        self.power_transform = power_transform
        self.use_sparse = use_sparse
        self.store_adjacency_matrix = store_adjacency_matrix
        self.padding = padding
        self.detrend = detrend
        self.metric = metric
        self.scale = scale
        self.aggregation_order = float(aggregation_order)
        self.fill_nan = fill_nan
        self.verbose = verbose
        
        np.random.seed(self.random_state)

    def _fillna(self, X):
        """
        Fill NaN values in the input time series using forward filling

        Args:
            X (array-like): A time series of shape (N, d) or (N,)

        Returns:
            X_filled (array-like): A time series of shape (N, d) or (N,) with NaN values
                filled using forward filling
        """
        if has_pandas:
            Xc = pd.DataFrame(X)
            Xc = Xc.fillna(method="ffill")
            Xc = Xc.fillna(method="bill")
            X_filled = Xc.values
        else:
            warnings.warn("Install pandas to use NaN filling")
            X_filled = np.copy(X)

        return X_filled

    def _make_embedding(self, X):
        """
        Create a time delay embedding of an input series. If the series is multivariate,
        create a multivariate embedding

        Args:
            X (array-like): A time series of shape (N, d) or (N,)

        Returns:
            X_embed (array-like): A time delay embedding of shape (N, d, d_embed)
        """
        np.random.seed(self.random_state)
        
        if len(X.shape) == 2:
            X = np.reshape(X, (X.shape[0], -1))
            X_embed = embed_ts(X, self.d_embed, padding=self.padding)
        elif len(X.shape) == 3:
            ## Multivariate
            warnings.warn("Multivariate time series detected, embedding each "
                          + "dimension separately."
                         )
            all_embeddings = list()
            for i in X.shape[-1]:
                X = np.reshape(X[..., i], (X.shape[0], -1))
                all_embeddings.append(
                    embed_ts(
                        np.reshape(X[..., i], (X.shape[0], -1)), 
                        self.d_embed, 
                        padding=self.padding
                    )
                )
            X_embed = np.hstack(all_embeddings)

        else:
            raise ValueError("Input shape not valid.")

        ## Regularization
        if self.noise > 0.0:
            X_embed += self.noise * np.random.normal(size=X_embed.shape)
            
        return X_embed
    
    def _preprocess(self, X):
        """
        Preprocess the input dataset using a series of transformations

        Args:
            X (np.ndarray): The input time series

        Returns:
            X (np.ndarray): The preprocessed time series
        """
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
        """
        Find a the distance matrix across an ensemble

        Args:
            X (np.ndarray): The input time series
        
        Development: either parallelize or batch depending on size
        """
        all_dist_mat = np.sqrt(
            np.sum((X[..., None, :] - X[:, None, ...]) ** 2, axis=-1)
        ).T
        all_dist_mat += (np.eye(all_dist_mat.shape[0]) * 1e16)[..., None]
        return all_dist_mat

    def _neighbors_to_cliques(self, bdmat):
        """
        Given a binary neighbor matrix, convert to a clique graph

        Args:
            bdmat (np.ndarray): binary adjacency matrix

        Returns:
            clique_matrix (np.ndarray): binary adjacency matrix with augmented cliques
        """
        #print("asymmetry: ", np.sum(np.abs(bdmat.T - bdmat)))
        clique_matrix = adjmat_from_associations(
            hollow_matrix(bdmat),
            weighted=self.weighted_connectivity,
            use_sparse=self.use_sparse,
        )
        return clique_matrix

    def fit_transform(self, X, y=None):
        """
        Args:
            X (array-like): A matrix of shape (n_timepoints, n_features)
            y (ignored) : Not used, present here for consistency with sklearn API
        """
        return self.fit_predict(X, y)

from sklearn.manifold import Isomap
class HirataNomuraIsomap(RecurrenceModel):
    """
    An implementation of the recurrence manifold model of Hirata et al. (2008) using
    the consensus similarity matrix of Nomura et al. (2022).

    Parameters:
        percentile (float): The percentile of the distance matrix to use as a threshold
            for determining whether two nodes are neighbors.
        n_components (int): The number of components to use in the Isomap embedding.
    """
    def __init__(self, n_components=2, percentile=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.percentile = percentile
    
    def fit(self, X):
        np.random.seed(self.random_state)
        X0 = np.copy(X)
        X = self._preprocess(X)
        X = self._make_embedding(X)

        # Compute the distance matrix
        amat = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            dmat = cdist(X[i], X[i])
            thresh = np.percentile(dmat, self.percentile)
            ## neighbor aggregation
            amat += (dmat <= thresh).astype(int)
        # Boolean aggregation
        amat[amat > 0] = 1

        wmat = common_neighbors_ratio(amat)
        if self.store_adjacency_matrix:
            self.adjacency_matrix = wmat

        iso = Isomap(n_components=self.n_components, metric='precomputed')
        pt_vals = iso.fit_transform(wmat)

        # Compare to spectral embedding
        # embedder = SpectralEmbedding(
        #     n_components=self.n_components, 
        #     random_state=self.random_state, 
        #     affinity='precomputed'
        # )
        # pt_vals = embedder.fit_transform(1 - wmat)

        self.indices = np.arange(len(pt_vals))
        self.labels_ = nan_fill(pt_vals)
        return self
    
    def transform(self, X):
        X = self._preprocess(X)
        X = self._make_embedding(X)
        wmat = common_neighbors_ratio(X)
        return wmat
    
    
class ClassicalRecurrenceClustering(RecurrenceModel):
    """
    A classical implementation of recurrence clustering, using the original 
    equivalence class algorithm of Sauer (PRL 2004)

    This class clusters time series data by constructing a binary distance
    matrix and finding the clusters in the data using a union-find algorithm on a
    connectivity graph constructed from the distance matrix.

    Attributes:
        adjacency_matrix: The adjacency matrix constructed from the input data.
            This attribute is only populated if the `store_adjacency_matrix`
            parameter is set to `True` when initializing the class instance.
        indices: The indices of the input data points that were used to
            construct the adjacency matrix.
        labels_: The labels assigned to each data point, indicating the
            cluster to which it belongs.    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(
        self, X, weighted=False, use_sparse=False,
    ):
        """
        Clusters the input time series data.

        Args:
            X: The time series data to be clustered, as a 2D NumPy array
                with shape (n_samples, n_features).
            weighted: Whether to use weighted connectivity when
                constructing the adjacency matrix.
            use_sparse: Whether to use a sparse representation for the
                adjacency matrix.

        Returns:
            None
        """
        X = self._preprocess(X)
        X = self._make_embedding(X)

        dist_mat_bin = data_to_connectivity(X, 
                                time_exclude=0,
                                use_sparse=False,
                                ord=500.,
                                scale=self.scale
                                )
        dist_mat_bin = sparsify(dist_mat_bin, (1 - self.tolerance), weighted=self.weighted_connectivity)
        
        if self.store_adjacency_matrix:
            self.adjacency_matrix = dist_mat_bin

        all_merged_inds = list()
        for row in dist_mat_bin:
            all_merged_inds.append(np.sort(np.where(row)[0]))

        merged_inds = solve_union_find([list(item) for item in all_merged_inds])
        merged_inds = [np.sort(np.array(item)) for item in merged_inds]
        indlist_lens = np.array([len(item) for item in merged_inds])

        known_items = list()
        item_labels = list()

        for item in merged_inds:
            i = 0
            add_flag = False
            for j, known_item in enumerate(known_items):
                if allclose_len(item, known_item):
                    item_labels.append(j)
                    add_flag = True
            if not add_flag:
                known_items.append(item)
                item_labels.append(i + 1)
        item_labels = np.array(item_labels)

        reference_indices = np.arange(dist_mat_bin.shape[0])
        #         missing_vals = np.setxor1d(reference_indices, indices)
        #         unclassified_inds = arg_find(reference_indices, missing_vals)
        self.indices = np.copy(reference_indices)
        self.labels_ = item_labels


class RecurrenceClustering(RecurrenceModel):
    """
    Assign a discrete set of labels to points in a time series, based on community 
    structure in the recurrence network. This model works best for discrete time
    dynamical systems.

    Attributes:
        resolution (float): The resolution parameter for the Leiden clustering algorithm
        labels_ (np.ndarray): cluster labels for each timepoint

    """

    def __init__(self, resolution=1.0, **kwargs):
        super().__init__(**kwargs)
        self.resolution = resolution

    def fit(
        self, X, objective="modularity", method="graspologic", use_sparse=False,
    ):
        """
        Args:
            X (array-like): A matrix of shape (n_timepoints, n_features)
            objective (str): The objective function to use for clustering.
                Options are "modularity" and "significance"
            method (str): The method to use for clustering. Options are "graspologic"
                and "leiden"
            use_sparse (bool): Whether to use sparse matrices for the Leiden algorithm

        """
        np.random.seed(self.random_state)
            
        X = self._preprocess(X)
        X = self._make_embedding(X)
        nbatch, ntime, ndim = X.shape
        
        dist_mat_bin = data_to_connectivity(X, 
                            time_exclude=self.time_exclude,
                            use_sparse=self.use_sparse,
                            ord=self.aggregation_order,
                            scale=self.scale
                            )
        #  dist_mat_bin =  = data_to_connectivity2(X)

        dist_mat_bin = sparsify(dist_mat_bin, (1 - self.tolerance), weighted=self.weighted_connectivity)
        #neighbor_matrix = self._neighbors_to_cliques(dist_mat_bin)
        neighbor_matrix = dist_mat_bin 

        if self.store_adjacency_matrix:
            self.adjacency_matrix = neighbor_matrix

        indices, labels = _leiden(
            neighbor_matrix,
            resolution=self.resolution,
            objective=objective,
            method=method,
            random_state=self.random_state,
        )
        sort_inds = np.argsort(indices)
        indices, labels = indices[sort_inds], labels[sort_inds]
        reference_indices = np.arange(neighbor_matrix.shape[0])

        self.indices = np.copy(reference_indices)
        self.labels_ = -np.ones_like(self.indices)
        self.labels_[indices] = labels

        self.has_unclassified = np.any(self.labels_ < 0)
        self.n_clusters = len(np.unique(self.labels_)) - self.has_unclassified


class RecurrenceManifold(RecurrenceModel):
    """
    Assign continuous time labels to a set of timepoints by finding recurrence families
     across multiple time series measurements
    """
    def __init__(self, n_components=1, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components

    def fit(
        self, X, use_sparse=False, root_index=0
    ):
        """
        Args:
            X (array-like): A matrix of shape (n_timepoints, n_features)      
            use_sparse (bool): Whether to use sparse distance matrices (appropriate for
                very large time series.
            root_index (int): The index of the first point in the drive signal.
        """
        np.random.seed(self.random_state)
        X0 = np.copy(X)
        X = self._preprocess(X)
        X = self._make_embedding(X)
        

        if self.verbose:
            print("Computing distance matrix... ", flush=True, end='')
            
        # Slowest step: compute the distance matrix for each example      
        ## an alternative approach that dodges the optimizer
        t1 = datetime.now()

        bd = data_to_connectivity(X, 
                            time_exclude=self.time_exclude,
                            ord=self.aggregation_order,
                            scale=self.scale
                            )
        # bd = data_to_connectivity2(X)
        t2 = datetime.now()
        elapsed = t2 - t1
        if self.verbose:
            print(f"Done in {elapsed.total_seconds():.2f} seconds", flush=True)

        neighbor_matrix = bd

        # self.uncompressed_matrix = neighbor_matrix.copy()            
        # neighbor_matrix = matrix_lowrank(neighbor_matrix, 1)
        # root_index = np.argmin(np.min(neighbor_matrix, axis=1))
        # self.root_index = root_index
        
        # rescale
        # neighbor_matrix = (neighbor_matrix - np.min(neighbor_matrix) + 1e-6) / (np.max(neighbor_matrix) - np.min(neighbor_matrix) + 1e-6)
        if self.verbose:
            print("Computing diffusion components... ", flush=True, end='')
        t1 = datetime.now()
        if self.store_adjacency_matrix:
            self.adjacency_matrix = neighbor_matrix

        # eigvals, eigvecs = scipy.linalg.eigh(
        #     neighbor_matrix, 
        #     subset_by_index=[neighbor_matrix.shape[0]-2, neighbor_matrix.shape[0]-1]
        # )
        # pt_vals = eigvecs[:, -1]

        # embedder = SpectralEmbedding(
        #     n_components=self.n_components, 
        #     random_state=self.random_state, 
        #     affinity='precomputed'
        # )
        # pt_vals = embedder.fit_transform(neighbor_matrix).squeeze()

        svd = TruncatedSVD(n_components=(self.n_components + 1))
        svd.fit(neighbor_matrix)
        pt_vals =svd.components_.T[:, 1:].squeeze()
        t2 = datetime.now()
        elapsed = t2 - t1
        if self.verbose:
            print(f"Done in {elapsed.total_seconds():.2f} seconds", flush=True)

        self.indices = np.arange(len(pt_vals))
        self.labels_ = nan_fill(pt_vals)


