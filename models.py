import numpy as np
import scipy
import warnings

from utils import *
from utils import embed_ts, standardize_ts, minmax_ts, find_psd
    

class DisjointSet:
    """
    A disjoint set data structure. 
    
    Adapted fro open-source code:
    https://stackoverflow.com/questions/67805907
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


def solve(lists):
    disjoint = DisjointSet()
    for lst in lists:
        disjoint.union(*lst)

    groups = disjoint.groups()
    return [lst and groups[disjoint.find(lst[0])] for lst in lists]


# def _embed(X, m, padding=None):
#     """
#     Create a time delay embedding of a time series or a set of time series

#     Args:
#         X (array-like): A matrix of shape (n_timepoints, n_dims) or 
#             of shape (n_timepoints)
#         m (int): The number of dimensions

#     Returns:
#         Xp (array-like): A time-delay embedding
#     """
#     if padding:
#         if len(X.shape) == 1:
#             X = np.pad(X, [m, 0], padding)
#         if len(X.shape) == 2:
#             X = np.pad(X, [[m, 0], [0, 0]], padding)
#         if len(X.shape) == 3:
#             X = np.pad(X, [[0, 0], [m, 0], [0, 0]], padding)
#     Xp = hankel_matrix(X, m)
#     Xp = np.moveaxis(Xp, (0, 1, 2), (1, 2, 0))
#     return Xp

import scanpy as sc
def find_pseudotime(dmat, root, n_branchings=0, fill=None, n_comps=15):
    """
    Compute pseudotime from a distance matrix
    
    Currently requires scanpy to be installed
    
    Args:
        dmat (array-like): a distance matrix of shape (N, N)
        root (integer): the index to use as the root
        n_branchings (int): the expected number of bifurcations
        fill (float or None): fill value for points unassigned pseudotime
        n_comps (int): the number of components to use in the diffusion map
        
    Returns
        pt_vals (array): a list of pseudotime assignments of length N
    """
    ndim = dmat.shape[0]
    adata = sc.AnnData(np.zeros((ndim, 3)), dtype='float64')
    sc.pp.neighbors(adata, n_neighbors=3, n_pcs=3)  ## no effect
    # this distance matrix is not used
    adata.obsp["distances"] = scipy.sparse.csr_matrix(dmat.shape)
    if scipy.sparse.issparse(dmat):
        adata.obsp["connectivities"] = dmat
    else:
        adata.obsp["connectivities"] = scipy.sparse.csr_matrix(dmat)
    adata.uns["iroot"] = root
    sc.tl.diffmap(adata, n_comps=n_comps) # Diffusion map calculation
    sc.tl.dpt(adata, n_branchings=n_branchings) # find maximum spanning tree
    pt_vals = np.array(adata.obs["dpt_pseudotime"])
    pt_vals[np.isinf(pt_vals)] = fill
    return pt_vals


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
from scipy.stats import iqr

try:
    from tslearn.metrics import cdist_dtw
except ImportError:
    has_tslearn = False

def data_to_connectivity(X, return_extremum=False, merge="percentile", 
                         time_exclude=0, use_sparse=None, scale=1.0, metric="euclidean"):
    """
    Given a stack of M time series, each of shape N x D, compute a 
    single consolidated N x N connectivity or adjacency matrix
    
    Args:
        X (array-like): A list of coordinates of shape M x N x D, where M is the number 
            of time series, N is the number of time points, and D is their 
            dimensionality
        return_extremum (bool): Return the index of the extremum of the timepoints
        metric ("euclidean" | "dtw"): The metric to use for distance computation
    """

    # if metric == "dtw" and has_tslearn: cdist = cdist_dtw
    
    sel_inds = np.all(np.isclose(X, X[:, :1, :], 1e-12), axis=(1, 2))
    if np.sum(sel_inds) > 0:
         warnings.warn(f"{np.sum(sel_inds)} Constant time series detected. " 
                        + "Skipping these datasets.")
    X = X[np.logical_not(sel_inds)]
    nb, nt, nd = X.shape
    
    extrema_ranks = np.zeros(X.shape[1])
    
    thresh =  np.median(
        np.linalg.norm(X - np.median(X, axis=1, keepdims=True), axis=-1)
    )
    thresh = 1 / (scale * thresh)
    #print(thresh, "\n-------\n")
    
    ## Iterate over all time series, compute distance matrix
    # dstack, bdstack = np.zeros((2, nb, nt, nt))
    if use_sparse:
        bd = scipy.sparse.csr_matrix((nt, nt))
    else:
        bd = np.zeros((nt, nt))
    for i in range(nb):
        
        if not use_sparse:
            dmat = cdist(X[i], X[i]) # pairwise distance matrix
            ## Normalize and exponentiate to find connectivity matrix then mean merge
            surprise = dmat / np.std(dmat)
            #thresh = 1.0
            bd += (1 / nb) * np.exp(-surprise / thresh) 
        else:
            a = X[i]
            kd = KDTree(a)
            dists, _ = kd.query(a, k=5)
            dists = dists[:, 1:] # distance to k-nearest neighbors excluding self
            
            thresh = np.percentile(dists[dists > 0], 10)
            #print(thresh)
            ## cheat with exact threshold of dense matrix
            # thresh = np.percentile(np.ravel(cdist(X[i], X[i])), 5)
            
            # zero everything greater than threshold
            dmat = kd.sparse_distance_matrix(kd, 
                                             thresh, 
                                             p=2.0, 
                                             output_type='coo_matrix')
            #print(sparsity(dmat))
            #print(np.unique(np.ravel(dmat.toarray())))
            dmat =  scipy.sparse.csr_matrix(dmat)
            nz = dmat.nonzero()
            mn_mat = dmat[nz].mean()
            std_mat = np.sqrt((dmat.multiply(dmat))[nz].mean() - mn_mat**2)
            if std_mat < 1e-10:
                std_mat = 1.0
            dmat[nz] = - (dmat[nz] / std_mat) / thresh # negative zscore
            dmat.setdiag(0)
            dmat = dmat.expm1()
            dmat.data += 1
            dmat.data *= (1 / nb)
            dmat.setdiag(0)
            bd += dmat
            
            if np.any(np.isnan(dmat.toarray())):
                print(thresh, sparsity(dmat))
        if return_extremum:
            # update average distance list
            ave_dists = hollow_matrix(dmat).mean(axis=0)
            extrema_ranks += ave_dists / nb
    
    # remove adjacent timescales by zeroing near-diagonals in distance matrix
    if time_exclude > 0:
        mask_mat = np.ones_like(bd)
        mask_mat = 1 - (
            np.triu(mask_mat, k=-time_exclude)
            * np.tril(mask_mat, k=time_exclude)
        )
        bd *= mask_mat.astype(float)
    
    if return_extremum:
        extremum_index = np.argmin(extrema_ranks)
        return extremum_index, bd
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

from scipy.optimize import minimize_scalar, root_scalar
from scipy.stats import boxcox
import warnings

# from models import embed_ts
# from models import *


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
        box_cox (bool): Whether to apply a Box-Cox transformation to the input time series
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
        box_cox=False,
        weighted_connectivity=True,
        merge="min",
        use_sparse=False,
        store_adjacency_matrix=False,
        detrend=False,
        metric="euclidean",
        scale=1.0,
        padding="symmetric"
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
        self.box_cox = box_cox
        self.use_sparse = use_sparse
        self.store_adjacency_matrix = store_adjacency_matrix
        self.padding = padding
        self.detrend = detrend
        self.metric = metric
        self.scale = scale
        
        np.random.seed(self.random_state)

    def _make_embedding(self, X):
        """
        Create a time delay embedding of an input series. If the series is multivariate,
        create a multivariate embedding
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
        Preprocess the input dataset
        """
        if self.detrend:
            X = detrend_ts(X)
        
        if self.box_cox:
            X = detrend_ts(X)
            X = standardize_ts(X)
            X = X[:, np.std(X, axis=0) > 0]
            X = minmax_ts(X) + 0.01
            X = np.vstack([boxcox(item)[0] for item in X.T]).T
            #X = np.reshape(boxcox(np.ravel(X))[0], X.shape)
            X = detrend_ts(X)
            X = standardize_ts(X)
            
        if self.standardize:
            X = standardize_ts(X)
        
        return X

    def _find_distance_matrix(self, X):
        """
        Find a the distance matrix across an ensemble
        
        Development: either parallelize or batch depending on size
        """
        #print("distance shape:", X.shape, flush=True)
        all_dist_mat = np.sqrt(
            np.sum((X[..., None, :] - X[:, None, ...]) ** 2, axis=-1)
        ).T
        all_dist_mat += (np.eye(all_dist_mat.shape[0]) * 1e16)[..., None]
        return all_dist_mat

    def _flatten_distance_matrix(self, all_dist_mat, scaled=True):
        """
        Given a stack of distance matrices, flatten to a single consolidated 
        distance matrix.
        """
        all_dist_mat /= np.mean(all_dist_mat, axis=(0, 1), keepdims=True)
        if self.merge == "min":
            dist_mat_accumulated = np.min(all_dist_mat, axis=-1)
        elif self.merge == "kmin":
            k = max(1, int((1 - self.tolerance) * all_dist_mat.shape[-1]))
            dist_mat_accumulated = np.mean(
                np.sort(all_dist_mat, axis=-1)[..., :k], axis=-1
            )
        elif self.merge == "connectivity":
            conn = distance_to_connectivity(all_dist_mat)
            dist_mat_accumulated = np.mean(conn * all_dist_mat, axis=-1)
        elif self.merge == "mean":
            dist_mat_accumulated = np.mean(all_dist_mat, axis=-1)
        elif self.merge == "median":
            dist_mat_accumulated = np.median(all_dist_mat, axis=-1)
        elif self.merge == "hmean":
            dist_mat_accumulated = 1 / np.mean(1 / all_dist_mat, axis=-1)
        elif self.merge == "exp_mean":
            dist_mat_accumulated = -np.log(np.mean(np.exp(-all_dist_mat), axis=-1))
        elif self.merge == "percentile":
            dist_mat_accumulated = np.percentile(
                all_dist_mat, 100 * self.tolerance, axis=-1
            )
        else:
            warnings.warn("Reduction method not recognized, falling back to minimum.")
            dist_mat_accumulated = np.min(all_dist_mat, axis=-1)
#         print(
#             "asymmetry acc: ",
#             np.sum(np.abs(dist_mat_accumulated.T - dist_mat_accumulated)),
#         )
        return dist_mat_accumulated

    def _threshold_matrix(self, d):
        """
        Binarize a distance matrix or stack of distance matrices
        """
        # print("asymmetry: ", np.sum(np.abs(d.T -  d)))
        if self.weighted_connectivity:
            d = distance_to_connectivity(d, sparsity=(1 - self.tolerance))

        # eps = self.eps
        #         eps = np.percentile(np.ravel(d), 100 * self.tolerance)
        #         print(f"Selected threshold is {eps}")

        #         dist_mat_bin = np.copy(d < eps).astype(int)
        #         print(f"Distance matrix sparsity is is {sparsity(dist_mat_bin)}")

        if len(d.shape) < 3:
            dist_mat_bin = sparsify(d, sparsity=(1 - self.tolerance)).astype(int)
        # stack sparsity
        else:
            dist_mat_bin = np.dstack(
                [
                    # sparsify(d[..., i], (1 - d.shape[-1] * self.tolerance)).astype(int)
                    sparsify(d[..., i], (1 - self.tolerance)).astype(int)
                    for i in range(d.shape[-1])
                ]
            )

        if self.time_exclude > 0:
            mask_mat = np.ones_like(dist_mat_bin)
            mask_mat = 1 - (
                np.triu(mask_mat, k=-self.time_exclude)
                * np.tril(mask_mat, k=self.time_exclude)
            )
            dist_mat_bin *= mask_mat.astype(int)

        return dist_mat_bin

    def _flatten_binary_matrix(self, bdmat):
        """
        Given a stack of binary matrices, flatten along the last dimension
        """
        n = bdmat.shape[-1]
        cmat = np.sum(bdmat, axis=-1)
        cmat = sparsify(cmat, 1 - self.tolerance)
        return cmat

    def _neighbors_to_cliques(self, bdmat):
        """
        Given a binary neighbor matrix, convert to a clique graph
        """
        #print("asymmetry: ", np.sum(np.abs(bdmat.T - bdmat)))
        clique_matrix = adjmat_from_associations(
            hollow_matrix(bdmat),
            weighted=self.weighted_connectivity,
            use_sparse=self.use_sparse,
        )
        return clique_matrix
    
class ClassicalRecurrenceClustering(RecurrenceModel):
    """
    A classical implementation of recurrence clustering, using the original 
    equivalence class algorithm of Sauer (PRL 2004)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(
        self, X, weighted=False, use_sparse=False,
    ):
        X_embed = self._make_embedding(X)
        distance_matrix_stack = self._find_distance_matrix(X_embed)
        distance_matrix = self._flatten_distance_matrix(distance_matrix_stack)
        dist_mat_bin = self._threshold_matrix(distance_matrix)

        all_merged_inds = list()
        for row in dist_mat_bin:
            all_merged_inds.append(np.sort(np.where(row)[0]))

        merged_inds = solve([list(item) for item in all_merged_inds])
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

        reference_indices = np.arange(distance_matrix.shape[0])
        #         missing_vals = np.setxor1d(reference_indices, indices)
        #         unclassified_inds = arg_find(reference_indices, missing_vals)
        self.indices = np.copy(reference_indices)
        self.labels_ = item_labels


from models import _leiden
class RecurrenceClustering(RecurrenceModel):
    """
    Cluster timepoints in a set of time series using recurrence families across multiple 
    time series measurements
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
        """
        np.random.seed(self.random_state)
            
        X = self._preprocess(X)
        X = self._make_embedding(X)
        nbatch, ntime, ndim = X.shape
        
        dist_mat_bin = data_to_connectivity(X, 
                          return_extremum=False, 
                          time_exclude=self.time_exclude,
                          use_sparse=self.use_sparse,
                          scale=self.scale
                         )
       
        #print("test12", flush=True)
        dist_mat_bin = sparsify(dist_mat_bin, (1 - self.tolerance), weighted=self.weighted_connectivity)
        #neighbor_matrix = self._neighbors_to_cliques(dist_mat_bin)
        neighbor_matrix = dist_mat_bin 
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

        #         missing_vals = np.setxor1d(reference_indices, indices)
        #         unclassified_inds = arg_find(reference_indices, missing_vals)

        self.indices = np.copy(reference_indices)
        self.labels_ = -np.ones_like(self.indices)
        self.labels_[indices] = labels

        #         print(indices.shape, labels.shape)
        #         self.indices = indices
        #         self.labels_ = labels
        self.has_unclassified = np.any(self.labels_ < 0)
        self.n_clusters = len(np.unique(self.labels_)) - self.has_unclassified

    def get_driving(self, X):
        """
        Return the best estimate of the driving signal, using a maximum variance 
        heuristic
        
        Args:
            X (array-like): A matrix of shape (n_timepoints, n_features)
        
        DEV: switch to a heuristic that maximizes spacing
        """
        vals_recon = self._get_full_driving(X)

        # Select best cycle example using largest variance principle
        max_spread_inds = np.argmax(
            np.sum(np.var(vals_recon, axis=1), axis=-1), axis=-1
        )

        vals_recon = vals_recon[np.arange(len(max_spread_inds)), :, max_spread_inds]
        return vals_recon

    def _get_full_driving(self, X):
        """
        Return all possible estimates of the driving signal across system. Negative 
        indices are ignored
        
        Args:
            X (array-like): A matrix of shape (n_timepoints, n_features)
        
        Returns:
            class_example (np.ndarray): An array of shape (B, T, Te, D), where B is 
                the index of an input dataset for reconstruction, T is the effective 
                time index of the reconstructed signal. Te indexes particular points on 
                the driver. Each value of Te corresponds to a particular driver 
                representation.
        
        """

        X = np.reshape(X, (X.shape[0], -1))
        X = standardize_ts(X)  ## check this
        X_embed = embed_ts(X, self.d_embed)
        # (B, T, D)

        class_example = list()
        for label in np.unique(self.labels_):
            if label < 0:
                continue
            # class_example.append(X_embed[:, self.labels_==label, :][:, 0])
            class_example.append(X_embed[:, self.labels_ == label, :])
        time_cap = min([item.shape[1] for item in class_example])
        class_example = [item[:, :time_cap] for item in class_example]
        class_example = np.array(class_example)
        class_example = np.moveaxis(class_example, (0, 1, 2), (1, 0, 2))

        return class_example

    def transform(self, X):
        """
        Creates an embedding of a dataset based on the labels list
        
        Args:
            X (array-like): A matrix of shape (n_timepoints, n_features)
            
        """
        cycle_vals = self.get_driving(X)
        all_cycles = list()
        for i in range(cycle_vals.shape[0]):
            sel_inds = self.labels_[self.labels_ >= 0]
            all_cycles.append(cycle_vals[i][sel_inds])
        return np.array(all_cycles)

    def fit_transform(self, X, y=None):
        """
        Args:
            X (array-like): A matrix of shape (n_timepoints, n_features)
            y (ignored) : Not used, present here for consistency with sklearn API
        """
        return self.fit(X).transform(X)



from sklearn.decomposition import PCA
from scipy.stats import arcsine, beta
unique_unsorted = lambda a : a[np.sort(np.unique(a, return_index=True)[1])]
class RecurrenceManifold(RecurrenceModel):
    """
    Assign continuous time labels to a set of timepoints by finding recurrence families
     across multiple time series measurements
    """
    def __init__(self, 
                 start="multiple", 
                 n_samples_pseudotime=20, 
                 sampling_method_pseudotime="shifted_random", 
                 **kwargs
                ):
        super().__init__(**kwargs)
        self.start = start
        self.n_samples_pseudotime = n_samples_pseudotime
        self.sampling_method_pseudotime = sampling_method_pseudotime
        # np.random.seed(self.random_state)

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
        nbatch, ntime, ndim = X.shape
        
        # Slowest step: compute the distance matrix for each example      
        ## an alternative approach that dodges the optimizer
        curr_time() # slowest step here:
        print("Computing distance matrix... ", flush=True, end='')
        if self.start != "multiple":
            root_index, bd = data_to_connectivity(X, 
                                                  return_extremum=True, 
                                                  time_exclude=self.time_exclude,
                                                  use_sparse=self.use_sparse,
                                                  scale=self.scale
                                                 )
        else:
            bd = data_to_connectivity(X, 
                                      return_extremum=False, 
                                      time_exclude=self.time_exclude,
                                      use_sparse=self.use_sparse,
                                      scale=self.scale
                                     )
        curr_time()
        print("done.", flush=True)
        if self.start is not None:
            root_index = self.start
        
        if not self.use_sparse:
            ## Enforce a minimum sparsity level
            dist_mat_bin = sparsify(
                bd,
                1 - self.tolerance,
                weighted=self.weighted_connectivity
            )
        else:
            dist_mat_bin = bd
        curr_time()
        nt = dist_mat_bin.shape[0]
        print("Matrix sparsity is: ", sparsity(dist_mat_bin))


        ## Given a connectivity matrix, compute the neighbor graph and then reduce
        #curr_time()
        #neighbor_matrix = self._neighbors_to_cliques(dist_mat_bin)
        #neighbor_matrix = dist_mat_bin
        neighbor_matrix = bd
        #root_index = np.argmin(np.mean(neighbor_matrix, axis=1))
        root_index = np.argmin(np.min(neighbor_matrix, axis=1))
        self.root_index = root_index
        
        # rescale
        # neighbor_matrix = (neighbor_matrix - np.min(neighbor_matrix) + 1e-6) / (np.max(neighbor_matrix) - np.min(neighbor_matrix) + 1e-6)
        curr_time()
        if self.store_adjacency_matrix:
            self.adjacency_matrix = neighbor_matrix
        
        print("Assigning pseudotime labels.", flush=True)
        if self.start == "multiple":
            n_sample = min(self.n_samples_pseudotime, ntime - ndim)
            np.random.seed(self.random_state)
            if self.sampling_method_pseudotime == "random":
                start_inds = np.random.choice(np.arange(ntime - ndim), n_sample, replace=False)
            if self.sampling_method_pseudotime == "random_extrema":
                sel_inds = (arcsine.rvs(size=8*n_sample) * (ntime - ndim - 1)).astype(int)
                #sel_inds = (beta(2, 2).rvs(size=8*n_sample) * (ntime - ndim - 1)).astype(int)
                sel_inds = unique_unsorted(sel_inds)[:n_sample]
                start_inds =  np.argsort(np.mean(neighbor_matrix, axis=0))[sel_inds]
            elif self.sampling_method_pseudotime == "deterministic":
                start_inds = np.linspace(0, ntime - ndim, n_sample).astype(int)
            elif self.sampling_method_pseudotime == "deterministic_extreme":
                print("test", flush=True)
                start_inds = np.argsort(np.min(neighbor_matrix, axis=1))[:n_sample].astype(int) # could try percentile
                
                xi, _ = np.unravel_index(np.argsort(np.ravel(neighbor_matrix)), (neighbor_matrix.shape))
                start_inds = xi[:n_sample].astype(int)
                # start_inds = np.argsort(np.max(hollow_matrix(neighbor_matrix), axis=1))[-n_sample:].astype(int) # percentile
                # start_inds = np.hstack([
                #     np.argsort(np.min(neighbor_matrix, axis=1))[:n_sample // 2].astype(int), 
                #     np.argsort(np.max(hollow_matrix(neighbor_matrix), axis=1))[-n_sample // 2:].astype(int)
                #     ])
                start_inds = np.argsort(outlier_detection_pca(X0, cutoff=0.95))[:self.n_samples_pseudotime]
            elif self.sampling_method_pseudotime == "shifted_random":
                interval = (ntime - ndim) // n_sample
                shifts = np.random.choice(np.arange(interval), n_sample)
                start_inds = np.linspace(0, ntime - ndim - interval, n_sample).astype(int)
                start_inds += shifts
            else:
                warnings.warn("Sampling method not recognized, defaulting to random.")
                start_inds = np.random.choice(
                    np.arange(ntime - ndim), 
                    n_sample, 
                    replace=False
                )
            start_inds = np.append(start_inds, root_index) ## root finding heuristic
            

            all_labels = list()
            for ind in start_inds:
                pt_vals = find_pseudotime(neighbor_matrix, ind)             
                all_labels.append(pt_vals)
                print(".", end="", flush=True)
            all_labels = np.array(all_labels)
            all_labels = np.array([nan_fill(item) for item in all_labels]) ##

            # weights: pseudotime uniformity
            # jitter_vals = np.var(np.diff(np.sort(all_labels, axis=1), axis=1), axis=1)
            even_spacing = np.linspace(0, 1, all_labels.shape[1])
            jitter_vals = np.sum(
                (np.sort(all_labels, axis=1) - even_spacing[None, :])**2,
                axis=1
            )
            data_weights = 1 / (jitter_vals + 1e-6)
            data_weights = data_weights / np.sum(data_weights)
            data_weights = None


            ## pick slowest labeling to avoid artifacts of bad starting point
            all_freqs = list()
            for labels in all_labels:
                freqs, wgts = find_psd(nan_fill(labels))
                all_freqs.append(np.sum(freqs * wgts) / np.sum(wgts))
            

            
            #weights = nan_pca(all_labels.T, weights=data_weights)[0]
            #pt_vals = np.dot(all_labels.T, weights)

            pt_vals = PCA(whiten=True).fit_transform(all_labels.T)[:, 0]

            ### ADDED
            self.all_labelings = all_labels

            ## pick slowest labeling to avoid artifacts of bad starting point
            # sel_ind = np.argmin(all_freqs)
            # pt_vals = all_labels[sel_ind]

            ## pick max norm labeling
            #sel_ind = np.argmax(np.nanmean(all_labels, axis=1))

            ## pick jitter labeling
            # sel_ind =  np.argmin(
            #     np.mean(np.abs(np.diff(all_labels, axis=1)), axis=1)
            # )
            
            
            #sel_ind = np.argmin(jitter_vals)


            # sel_ind = np.argmin(gini_skew) ## can take weighted average as well
                    
            #pt_vals = all_labels[sel_ind]
        else:
            pt_vals = find_pseudotime(neighbor_matrix, root_index)
        print("\n")
        self.indices = np.arange(len(pt_vals))
        self.labels_ = nan_fill(pt_vals)

