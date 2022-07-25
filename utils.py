import numpy as np
import networkx as nx

## Array and list utilities

from scipy.linalg import hankel

from scipy.signal import blackmanharris, periodogram

from datetime import datetime


def get_time():
    """Find current time in human-readable format"""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time
    
def curr_time():
    """Print current time"""
    print("Current Time: ", get_time(), flush=True)
    
def nan_pca(X, weights=None):
    """
    Perform pca on a data matrix with missing values

    Args:
        X (array): A data matrix with shape (N, D)
        weights (array): A weight matrix with shape N,
    """
    print("wPCA", X.shape, flush=True)
    if weights is None:
        weights = np.ones(X.shape[1])
    w_sum = np.nansum(weights)
    w_mat = np.diag(weights)
    X_mean = (1 / w_sum) * np.nansum(weights * X, axis=0, keepdims=True)
    X = X - X_mean
    print(X.shape, w_mat.shape)
    cov = (1 / w_sum) * np.nansum(X[..., None] * w_mat.dot(X.T).T[:, None, :], axis=0)
    eigs, vecs = np.linalg.eigh(cov)
    eigs, vecs = eigs[::-1], vecs[::-1]
    return vecs


def discretize_signal(signal, max_states=50):
    """
    Given a continuous signal, discretize into a finite number of states

    Args:
        signal (array): A signal with shape (T, D)
        max_states (int): The maximum number of states to use
    """
    n = len(signal)
    _, bins = np.histogram(signal, max_states)
    signal_d = np.digitize(signal, bins, right=True) # returns which bin
    
    ## prune bins with few elements
    vals, counts = np.unique(
        signal_d,
        return_counts=True
    )
    sel_vals = (counts / n) < 0.01
    for i, loc in enumerate(vals[sel_vals]):
        if i + 1 < len(vals):
            signal_d[signal_d == loc] = vals[i + 1]
        else:
            signal_d[signal_d == loc] = vals[i - 1]
    
    ## convert to fewer labels
    keys, vals = np.unique(signal_d), np.arange(len(signal_d))
    trans_dict = dict(zip(keys, vals))
    signal_d = np.array([trans_dict[key] for key in signal_d])
    
    return signal_d

def zero_topk(a, k=1, magnitude=False):
    """
    Return a copy of a vector with all components except the top k set equal to zero

    Args:
        a (array): A vector with shape (D,)
        k (int): The number of components to zero
        magnitude (bool): If True, zero everything but the top k components
    """
    a2 = np.zeros_like(a)
    if magnitude:
        topk_inds = np.argsort(np.abs(a))[::-1][:k]
    else:
        topk_inds = np.argsort(a)[::-1][:k]
    a2[topk_inds] = a[topk_inds]
    return a2
    
import pickle
def load_pickle_file(filename):
    """
    Load an unstructured pickle file
    """
    fr = open(filename, 'rb')
    data = pickle.load(fr)
    fr.close()
    return data

from scipy.signal import find_peaks
def discretize_ts(ts, **kwargs):
    """
    Given a univariate time series, return a discretization based on peak crossings
    
    keyword arguments are passed on to scipy.find_peaks
    """
    peak_inds = np.sort(np.hstack([find_peaks(-ts, **kwargs)[0], 
                                   find_peaks(ts, **kwargs)[0]]))
    return ts[peak_inds]

def detrend_ts(a, method="global_linear"):
    """
    Detrend a time series along its first axis using a variety of methods
    
    Arguments
        a (array): A time series of shape (T, D)
        method (str): "global_linear" - subtracts the best fit straight line from the data
                      "naive" - subtracts the line bridging the first and final values
    
    Development
        Fully vectorize line fitting by estimating inverse of Vandermonde matrix
    """
    if len(a.shape) < 2:
        a = a[:, None]
        
    if method == "naive":
        trend = a[0] + (a[-1] - a[0])[None, :] * np.arange(a.shape[0])[:, None]
        return a - np.squeeze(trend)
    elif method == "global_linear":
        all_trends = list()
        for row in a.T:
            m, b = np.polyfit(np.arange(a.shape[0]), row, 1)
            trend = (m * np.arange(a.shape[0]) + b)
            all_trends.append(trend)
        all_trends = np.array(all_trends).T
        return a - all_trends
    elif method == "global_exponential":
        all_trends = list()
        for row in a.T:
            m, b = np.polyfit(np.arange(a.shape[0]), np.log(row), 1)
            trend = np.exp(m * np.arange(a.shape[0])) * np.exp(b)
            all_trends.append(trend)
        all_trends = np.array(all_trends).T
        return a - all_trends
    else:
        trend = (a[-1] - a[0]) * np.arange(a.shape[0])
        return np.squeeze(a - trend)

def nan_fill(a):
    """
    Backfill nan values in a numpy array 
    """
    out = np.copy(a)
    for i in range(out.shape[0]):
        if np.isnan(out[i]):
            out[i] = out[i- 1]
    return out

def unroll_phase(theta0, wrapval=2*np.pi):
    """
    Given a list of phases, unroll them in order to prevent wraparound discountinuities
    """
    theta = np.copy(theta0)
    sel_inds = np.abs(np.diff(theta)) > wrapval * 0.9
    inds = np.sort(np.where(sel_inds)[0])
    for ind in inds:
        diffval = theta[ind + 1] - theta[ind]
        theta[ind+1:] -= np.sign(diffval) * wrapval
    return theta

def neighbors_to_mutual(a):
    """
    Given either a binary neighbor matrix or a connectivity matrix with all elements
    between zero and one, return a mutual neighbor matrix, which is a symmetric 
    matrix comprising a subgraph of mutual neighbors
    """
    return (a * a.T) / (0.5 * (a + a.T))
    

def arg_find(search_vals, target_vals):
    """
    Return the indices in search_vals pointing to the values in target_vals
    The output is unordered
    """
    search_vals, target_vals = np.asarray(search_vals), np.asarray(target_vals)
    return np.where(
        np.prod(search_vals[:, None] - target_vals[None, :], axis=1) == 0
    )[0]

def find_psd(y, window=True):
    """
    Find the power spectrum of a signal
    """
    if window:
        y = y * blackmanharris(len(y))
    halflen = int(len(y)/2)
    fvals, psd = periodogram(y, fs=1)
    return fvals[:halflen], psd[:halflen]

def lift_ts(a, target=2):
    """
    If needed, pad the dimensionality of a univariate time series
    """
    deficit = target - len(a.shape)
    if deficit == 1:
        return a[:, None]
    else:
        return a

from scipy.stats import iqr
def standardize_ts(a, scale=1.0, median=False):
    """Standardize an array along dimension -2
    For dimensions with zero variance, divide by one instead of zero
    
    Args:
        a (ndarray): a matrix containing a time series or batch of time series
            with shape (T, D) or (B, T, D)
        scale (float): the number of standard deviations by which to scale
        median (bool): whether to use median/IQR to normalize
    
    Returns:
        ts_scaled (ndarray): A standardized time series with the same shape as 
            the input
    """
    a = lift_ts(a)
    
    if median:
        center = np.median(a, axis=-2, keepdims=True)
        stds = iqr(a, axis=-2, keepdims=True)
    else:
        center = np.mean(a, axis=-2, keepdims=True)
        stds = np.std(a, axis=-2, keepdims=True)
    stds[stds==0] = 1
    ts_scaled = (a - center) / (scale * stds)
    return np.squeeze(ts_scaled)

def minmax_ts(a, clipping=None):
    """MinMax scale an array along dimension -2
    For dimensions with zero variance, divide by one instead of zero
    
    Args:
        a (ndarray): a matrix containing a time series or batch of time series
            with shape (T, D) or (B, T, D)
        clipping (float): A number between 0 and 1, the range of values 
            to use for rescaling
    
    Returns:
        ts_scaled (ndarray): A minmax scaled time series with the same shape as 
            the input
    """
    a = lift_ts(a)
    
    if clipping:
        minval = np.percentile(a, clipping * 100, axis=-2, keepdims=True)
        maxval = np.percentile(a, (1 - clipping) * 100, axis=-2, keepdims=True)       
    else:
        minval = np.min(a, axis=-2, keepdims=True)
        maxval = np.max(a, axis=-2, keepdims=True)
    spans = (maxval - minval)   
    spans[spans==0] = 1
    ts_scaled = (a - minval) / spans
    return np.squeeze(ts_scaled)

def embed_ts(X, m, padding=None):
    """
    Create a time delay embedding of a time series or a set of time series

    Args:
        X (array-like): A matrix of shape (n_timepoints, n_dims) or 
            of shape (n_timepoints)
        m (int): The number of dimensions

    Returns:
        Xp (array-like): A time-delay embedding
    """
    if padding:
        if len(X.shape) == 1:
            X = np.pad(X, [m, 0], padding)
        if len(X.shape) == 2:
            X = np.pad(X, [[m, 0], [0, 0]], padding)
        if len(X.shape) == 3:
            X = np.pad(X, [[0, 0], [m, 0], [0, 0]], padding)
    Xp = hankel_matrix(X, m)
    Xp = np.moveaxis(Xp, (0, 1, 2), (1, 2, 0))
    return Xp



def hankel_matrix(data, q, p=None):
    """
    Find the Hankel matrix dimensionwise for multiple multidimensional 
    time series
    
    Args:
        data (ndarray): An array of shape (N, T, 1) or (N, T, D) corresponding to a 
            collection of N time series of length T and dimensionality D
        q (int): The width of the matrix (the number of features)
        p (int): The height of the matrix (the number of samples)
        
    Returns:
        hmat (ndarray)

    """
    
    if len(data.shape) == 3:
        return np.stack([_hankel_matrix(item, q, p) for item in data])
    
    if len(data.shape) == 1:
        data = data[:, None]
    hmat = _hankel_matrix(data, q, p)    
    return hmat
    

def _hankel_matrix(data, q, p=None):
    """
    Calculate the hankel matrix of a multivariate timeseries
    
    Args:
        data (ndarray): T x D multidimensional time series
    """
    if len(data.shape) == 1:
        data = data[:, None]

    # Hankel parameters
    if not p:
        p = len(data) - q
    all_hmats = list()
    for row in data.T:
        first, last = row[-(p + q) : -p], row[-p - 1 :]
        out = hankel(first, last)
        all_hmats.append(out)
    out = np.dstack(all_hmats)
    return np.transpose(out, (1, 0, 2))[:-1]

def hollow_matrix(arr):
    """
    Set the diagonal of a matrix to zero
    """
    return arr * (1 - np.eye(arr.shape[0]))

def allclose_len(arr1, arr2):
    """Test whether all entries are close, and return False
    if different shapes"""
    close_flag = False
    try:
        close_flag = np.allclose(arr1, arr2)
    except ValueError:
        close_flag = False
    return close_flag

def array2d_to_list(arr):
    """Convert a 2D ndarray to a list of lists"""
    return [list(item) for item in arr]

def community_list_to_labels(community_list):
    """
    Given a list of community members, create a list of labels
    """
    all_labels = list()
    for ind, com in enumerate(community_list):
        for member in com:
            all_labels.append((member, ind))
    
    all_labels = sorted(all_labels, key=lambda x: x[0])
    return all_labels

# def get_all_pairs(ind_list):
#     """
#     Return all unique pairs from the list
#     """
#     all_pairs = list()
#     for ind in ind_list:
#         all_pairs += [(ind, ind2) for ind2 in ind_list if ind2 > ind]
#     return all_pairs



## NetworkX utilities

import networkx as nx
def largest_connected_component(g):
    """Return the scaled largest connected component of a graph."""
    n = g.number_of_nodes()
    giant = max(nx.connected_components(g), key=len)
    lcc = len(giant) / n
    return lcc

def susceptibility_smallcomponents(g):
    """Return the susceptibility of a graph based on the size of small components."""
    n = g.number_of_nodes()
    all_components = np.array([len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)])
    return np.sum(all_components[1:]**2) / n

def susceptibility_subleading(g):
    """Return the susceptibility of a graph based on the size of largest subleading 
    component."""
    n = g.number_of_nodes()
    all_components = [len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    if len(all_components) > 1:
        return all_components[1] / n
    else:
        return 0

def sort_graph(g):
    """
    Return a graph with sorted nodes
    """
    h = nx.Graph()
    h.add_nodes_from(sorted(g.nodes(data=True)))
    h.add_edges_from(g.edges(data=True))
    return h

from collections import Counter

def multigraph_to_weighted(g):
    """
    Convert a MultiGraph to a weighted graph
    """
    c = Counter(g.edges()) 
    for u, v, d in g.edges(data=True):
        d['weight'] = c[u, v]
    adj = nx.linalg.graphmatrix.adjacency_matrix(g).todense()
    adj = np.sqrt(adj)
    out_g = nx.Graph(adj)
    return out_g

import itertools

def graph_from_associations(mat, weighted=False):
    """
    Given an association matrix, create a graph of all non-zero elements occurring 
    within a row are assumed to be densely connected to eachother, forming a clique
    
    Example:
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 1]
    
    Corresponds to the edges
    1 -- 2
    0 -- 1
    1 -- 2
    """
    if weighted:
        g = nx.MultiGraph()
    else:
        g = nx.Graph()
    for ind, row in enumerate(mat):
        inds = np.where(row)[0]
        g.add_edges_from(itertools.combinations(inds, 2))
        
    if weighted:
        g = multigraph_to_weighted(g)
    return g

def adjmat_from_associations(mat, weighted=False, use_sparse=False):
    """
    Given an association matrix, create an adjacency matrix 
    representing a graph of cliques
    
    All non-zero elements occurring within a row  of the input
    matrix are assumed to be densely connected to eachother, 
    forming a clique in the output matrix
    
    Example:
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 1]
    --->
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
    
    Starting from the edges
    0 -- 1
    0 -- 2
    2 -- 2
    We add the additional edges
    0 -- 0
    1 -- 2
    """

    if use_sparse:
        row_inds = list()
        col_inds = list()
        vals = list()
        for row in mat:
            inds = np.where(row)[0]
            for ind_pair in itertools.combinations(inds, 2):
                row_inds += [ind_pair[0], ind_pair[1]]
                col_inds += [ind_pair[1], ind_pair[0]]
                vals += [1, 1]
        g_adj = sp.csr_matrix((vals, (row_inds, col_inds)), shape=mat.shape)

    # NOT IMPLEMENTED: Weighted
    #     g_adj = g_adj.todense()
    #     if not weighted:
    #         g_adj[g_adj > 0] = 1
    else:

        g_adj = np.zeros(mat.shape)
        for row in mat:
            inds = np.where(row)[0]
            for ind_pair in itertools.combinations(inds, 2):
                g_adj[ind_pair[0], ind_pair[1]] += 1
                g_adj[ind_pair[1], ind_pair[0]] += 1

        ## Is this step even needed?
        if not weighted:
            g_adj[g_adj > 0] = 1

    return g_adj

def graph_threshold(g, threshold=1.0):
    """
    Given a weighted graph, return an unweighted graph defined by 
    thresholding the edges
    """
    h = nx.Graph()
    for u, v, d in g.edges(data=True):
        if d["weight"] > threshold:
            h.add_edge(u ,v)
    return h

def adjmat(g):
    """
    Wrapper for networkx adjacency matrix
    """
    return nx.linalg.graphmatrix.adjacency_matrix(g)

def multigraph_to_weighted(g):
    """
    Convert a MultiGraph to a weighted graph
    """
    c = Counter(g.edges()) 
    for u, v, d in g.edges(data=True):
        d['weight'] = c[u, v]
    adj = nx.linalg.graphmatrix.adjacency_matrix(g).todense()
    adj = np.sqrt(adj)
    out_g = nx.Graph(adj)
    return out_g

import scipy
def sparsity(a):
    """Compute the sparsity of a matrix"""
    if scipy.sparse.issparse(a):
        sparsity = 1.0 - a.getnnz() / np.prod(a.shape)
    else:
        sparsity = 1.0 - (np.count_nonzero(a) / float(a.size))
    return sparsity

def otsu_threshold(data0):
    """
    Calculate the Otsu threshold of a dataset
    """
    data = np.ravel(np.copy(data0))
    n = len(data)
    #nbins = np.sqrt(n)
    nbins = int(round(1 + np.log2(n)))

    data = np.sort(data)[::-1]
    
    hist, bins = np.histogram(data, nbins)
    bins = bins[:-1] + (bins[1] - bins[0]) / 2 # center bins
    hist = hist.astype(float)
    hist /= np.sum(hist) # normalized

    bin_index, cross_var_highest = nbins // 2, -1
    for i in np.arange(1, nbins - 1):
        pleft, pright = np.sum(hist[:i]), np.sum(hist[i:])

        mean_left = np.sum(hist[:i] * bins[:i])  / pleft
        mean_right = np.sum(hist[i:] * bins[i:]) / pright

        cross_var = pleft * pright * (mean_left - mean_right) ** 2

        if cross_var_highest < cross_var:
            bin_index = i
            cross_var_highest = cross_var
            
    return bins[bin_index]


def sparsify(a0, sparsity=None, weighted=False):
    """
    Binarize a matrix by thresholding its values, resulting in a matrix with a given
    sparsity. 
    
    If the matrix contains duplicate elements, thresholding is performed 
    in such a way as to ensure the sparsity is *at least* the requested value
    
    Args:
        a0 (array-like): an array to binarize
        sparsity (float or None): the target fraction of zeros in the output array. If
            no sparsity is given, a threshold is calculated based on the Otsu method.
        weighted (bool): Whether to keep sparse-elements or zet them equal to one
    
    Returns
        a (array-like): A binary matrix
    """
    if sparsity is None:
        sparsity = otsu_threshold(a0)
    a = a0.copy()
    denom = np.sum(np.ones_like(a))
    thresh = np.percentile(np.ravel(np.abs(a)), 100 * sparsity, interpolation="higher")
    a[np.abs(a) <= thresh] = 0  # sparsify
    if weighted:
        pass
    else:
        a[np.abs(a) > thresh] = 1
    return a


from sklearn.decomposition import PCA
def outlier_detection_pca(X, cutoff=0.95):
    """
    Detect outliers in a dataset using PCA.

    Args:
        X (array): Dataset of shape (n_samples, n_features)
        cutoff (float): cutoff between 0 and 1 for variance used for outlier detection

    Returns:
        scores (array): scores of shape (n_samples,)

    """
    pca = PCA()
    wv = pca.fit_transform(X)
    #print(wv.shape)
    sel_inds = np.cumsum(pca.explained_variance_ratio_) < cutoff
    #print(sel_inds.shape)
    pc = pca.components_
    pc_truncated = pc[sel_inds]
    
    X_recon = np.dot(wv[:, sel_inds], pc[sel_inds])

    scores = np.linalg.norm(X - X_recon, axis=-1)
    return scores

## Baselines

import sklearn.metrics

def evaluate_clustering(labels_true, labels_pred):
    """
    Given a set of known and predicted cluster labels, compute a set of cluster quality 
    metrics that is invariant to permutations

    Args:
        labels_true (array): true cluster labels
        labels_pred (array): predicted cluster labels

    Returns:
        dict: A dictionary of cluster quality metrics
    """
    metric_names = ["rand_score",
                    "adjusted_rand_score",
                    "fowlkes_mallows_score", 
                    "normalized_mutual_info_score",
                    "adjusted_mutual_info_score",
                    "homogeneity_score", 
                    "completeness_score",
                    "v_measure_score"
                   ]
    recorded_metrics = dict()

    for metric_name in metric_names:
        metric_func = getattr(sklearn.metrics, metric_name)
        recorded_metrics[metric_name] = metric_func(labels_true, labels_pred)
        
    return recorded_metrics

