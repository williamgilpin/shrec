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
    
    Returns:
        X_transformed (array): The data after projection onto the eigenvectors of the 
            covariance matrix
    """
    if weights is None:
        weights = np.ones(X.shape[1])
    w_sum = np.nansum(weights)
    w_mat = np.diag(weights)
    X_mean = (1 / X.shape[0])* np.nansum(weights * X, axis=0, keepdims=True)
    X = X - X_mean
    cov = (1 / w_sum) * np.nansum(X[..., None] * w_mat.dot(X.T).T[:, None, :], axis=0)
    eigs, vecs = np.linalg.eigh(cov)
    vecs = vecs.T
    eigs, vecs = eigs[::-1], vecs[::-1]

    X_transformed = X.dot(vecs.T)
    return X_transformed

    # X = X - np.mean(X, axis=0, keepdims=True)
    # cov = np.dot(X.T, X)
    # #cov = np.cov(X.T)
    # print(cov)
    # eigs, vecs = np.linalg.eigh(cov)
    # vecs = vecs.T
    # eigs, vecs = eigs[::-1], vecs[::-1]
    # return vecs

    # X -= np.mean(X, axis = 0) 
    # cov = np.cov(X, rowvar = False)
    # eigs, vecs = np.linalg.eigh(cov)
    # idx = np.argsort(eigs)[::-1]
    # vecs = vecs[:,idx]
    # eigs = eigs[idx]
    # return vecs.T

def matrix_lowrank(a, k=-1):
    """Returns the low-rank approximation of a matrix"""
    U, s, V = np.linalg.svd(a)
    return U[:, :k] @ np.diag(s[:k]) @ V[:k, :]

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

    Returns:
        a2 (array): A copy of a with the top k components set equal to zero
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
    Load an unstructured pickle file and return the data

    Args:
        filename (str): The path to the pickle file

    Returns:
        data (object): The data stored in the pickle file
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

try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import kpss
    has_stat = True
except ImportError:
    has_stat = False

def transform_stationary(ts, pthresh=0.05):
    """ 
    Transform a time series to be stationary using the Augmented Dickey-Fuller test and
    the Kwiatkowski-Phillips-Schmidt-Shin test. Depending on which combination of tests
    the time series passes, the time series is transformed using differencing or
    detrending.

    Args:
        ts (np.ndarray): Time series to be transformed.
        pthresh (float): Threshold for p-value of statistical tests.

    Returns:
        out (np.ndarray): Transformed time series.
    """
    if not has_stat:
        warnings.warn("Statsmodels not found, install to use this function")
    ts = np.squeeze(ts).copy()
    ad_fuller = adfuller(ts)
    kpss_test = kpss(ts)

    if kpss_test[1] < pthresh:
        ts = np.diff(ts)

    if ad_fuller[1] > pthresh:
        ts = detrend_ts(ts)

    return ts.squeeze()

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

    Args:
        y (array): A signal of shape (T,)

    Returns:
        fvals (array): The frequencies of the power spectrum with shape (T // 2,)
        psd (array): The power spectrum of shape (T // 2,)

    """
    if window:
        y = y * blackmanharris(len(y))
    halflen = int(len(y)/2)
    fvals, psd = periodogram(y, fs=1)
    return fvals[:halflen], psd[:halflen]


def group_consecutives(vals, step=1):
    """
    Return list of consecutive lists of numbers from vals (number list).
    
    References:
        Modified from the following
        https://stackoverflow.com/questions/7352684/
        how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy 
    """
    run = list()
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

from scipy.ndimage import gaussian_filter1d
def find_characteristic_timescale(y, k=1, window=False):
    """
    Find the k leading characteristic timescales in a time series
    using the power spectrum..
    """
    y = gaussian_filter1d(y, 3)

    fvals, psd = find_psd(y, window=window)
    max_indices = np.argsort(psd)[::-1]
    
    # Merge adjacent peaks
    grouped_maxima = group_consecutives(max_indices)
    max_indices_grouped = np.array([np.mean(item) for item in grouped_maxima])
    max_indices_grouped = max_indices_grouped[max_indices_grouped != 1]
    
    return np.squeeze(1/(np.median(np.diff(fvals))*max_indices_grouped[:k]))

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


from sklearn.base import BaseEstimator, TransformerMixin

class RigidTransform(BaseEstimator, TransformerMixin):
    """
    Compute the rigid transformation (rotation, translation, scaling) aligning two
    sets of points using the Kabsch-Umeyama algorithm. The first argument is transformed
    to align with the second argument.

    Attributes:
        scale (float): scaling factor
        rotation (array): rotation matrix
        translation (array): translation vector
    """

    def __init__(self, scale=None, rotation=None, translation=None):
        self.scale = scale
        self.rotation = rotation
        self.translation = translation

    def fit(self, X, y=None):
        """
        Fit transformation matrices that align X to y.

        Args:
            X: (N, M) array of points to be transformed
            y: (N, M) array of points to align with
        """
        b, a = X, y
        assert a.shape == b.shape
        n, m = a.shape

        ca = np.mean(a, axis=0)
        cb = np.mean(b, axis=0)

        var_a = np.mean(np.linalg.norm(a - ca, axis=1) ** 2)

        H = ((a - ca).T @ (b - cb)) / n
        U, D, VT = np.linalg.svd(H)
        d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
        S = np.diag([1] * (m - 1) + [d])

        R = U @ S @ VT
        c = var_a / np.trace(np.diag(D) @ S)
        t = ca - c * R @ cb

        transform_params = (R, c, t)

        self.rotation = R
        self.scale = c
        self.translation = t

    def fit_transform(self, X, y=None):
        """
        Fit transformation matrices that align X to y and apply them to X.

        Args:
            X: (N, M) array of points to be transformed
            y: (N, M) array of points to align with
        """
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        X_transformed = self.scale * self.rotation @ X.T + self.translation[:, np.newaxis]
        return X_transformed.T

    def inverse_transform(self, X):
        X = X.T
        X = (X - self.translation[:, np.newaxis]) / self.scale
        X = self.rotation.T @ X
        return X.T


from sklearn.preprocessing import PowerTransformer

def make_surrogate(X, ns=1, method="random_phase", gaussianize=True, random_state=None):
    """
    Create a surrogate time series from a given time series. If gaussianize is True,
    then the surrogate will be gaussianized beforehand. The default configuration 
    approximates the AAFT surrogate method.
    
    Args:
        X (ndarray): A one-dimensional time series
        method (str): "random_shuffle" or "random_phase"
        gaussianize (bool): If True, the surrogate will be gaussianized
        random_state (int): Random seed for reproducibility
        
    Returns:
        Xs (ndarray): A single random surrogate time series
    """
    if gaussianize:
        model = PowerTransformer(standardize=True)
        X = np.squeeze(model.fit_transform(X[:, None]))

    np.random.seed(random_state)
    if method == "random_phase":
        phases, radii = np.angle(np.fft.fft(X)), np.abs(np.fft.fft(X))
        random_phases = 2 * np.pi * (2 * (np.random.random(size=(phases.shape[0], ns)) - 0.5))
        Xs = np.real(
            np.fft.ifft(
            radii[:, None] * np.cos(random_phases) + 1j * radii[:, None] * np.sin(random_phases),
            axis=0
            )
        )
    else:
        Xs = np.random.permutation(X)

    if gaussianize:
        Xs = np.array([
            model.inverse_transform(item[:, None]) for item in Xs.T
        ]).T

    Xs = np.squeeze(Xs)
    return Xs
    

def array_select(arr, inds):
    """
    Selects a subset of an array, given a set of indices or boolean slices
    """
    arr_out = np.copy(arr)
    arr_out = arr_out[inds]
    arr_out = arr_out[:, inds]
    return arr_out

def dict_to_vals(d):
    """
    Convert a dictionary to an array of values, in the order of the sorted keys.

    Args:
        d (dict): dictionary to convert

    Returns:
        d_arr (np.array): array of values
        key_names (list): list of keys
    """
    d_arr = list()
    key_names = sorted(d.keys())
    for key in key_names:
        d_arr.append(d[key])
    d_arr = np.array(d_arr)
    return d_arr, key_names

def spherize_ts(X):
    """Spherize a time series with PCA"""
    X = X - np.mean(X, axis=0)
    X = X.dot(PCA().fit(X).components_.T)
    return X

def whiten_zca(X):
    """
    Whiten a dataset with ZCA whitening (Mahalanobis whitening).
    Args:
        X: numpy array of shape (n_samples, n_features)
    """
    sigma = np.cov(X, rowvar=True)
    U, S, V = np.linalg.svd(sigma)
    zmat = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + 1e-6)), U.T))
    X_out = np.dot(zmat, X)
    return X_out
    

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



def merge_labels(labels_list, labels, label_merge="majority"):
    """
    Given a jagged list of labels, merge them into a single list of labels.

    Args:
        labels_list (list): A list of lists of labels.
        labels (list): A list of labels.
        label_merge (str): How to merge labels. Options are "majority" and "average".

    Returns:
        labels_consolidated (list): A list of labels for the merged network
    """

    labels_consolidated = list()
    for item in labels_list:
        if np.isscalar(item):
            labels_consolidated.append(labels[item])
        else:
            votes = list()
            for item2 in item:
                # votes.append(np.argmax(np.bincount(item2)))
                votes.append(labels[item2])

            if label_merge == "majority":
                consensus = np.argmax(np.bincount(votes))
                labels_consolidated.append(votes[consensus])
            elif label_merge == "average":
                labels_consolidated.append(np.mean(votes))
            else:
                consensus = np.argmax(np.bincount(votes))
                labels_consolidated.append(votes[consensus])

    return labels_consolidated

def compress_adjacency(amat0, n_target, return_labels=False, label_merge="majority"):
    """
    Consolidate a network by merging correlated nodes.

    Args:
        amat0 (np.ndarray): Adjacency matrix of the network to be compressed.
        n_target (int): Number of nodes desired for the output network
        labels (np.ndarray | None): If labels are passed, the labels are matched to the 
            compressed network by taking the most common label of the nodes that were 
            merged.

    Returns:
        amat (np.ndarray): Compressed adjacency matrix
        labels (np.ndarray): Labels of the compressed network
    
    """

    amat = np.copy(amat0)
    all_labels = np.arange(amat.shape[0])

    # drop unconnected components
    where_unconnected = np.isclose(np.sum(np.abs(amat), axis=0), 0)
    amat = amat[np.logical_not(where_unconnected)]
    amat = amat[:, np.logical_not(where_unconnected)]
    amat = hollow_matrix(amat)
    
    all_labels = all_labels[np.logical_not(where_unconnected)].tolist()
    

    n_steps = max(amat.shape[0] - n_target, 0)
    
    for step_ind in range(n_steps):

        if step_ind % (n_steps // 20) == 0:
            print(100 * step_ind / n_steps, end=" ")
        
        amat_m = amat - np.mean(amat, axis=0, keepdims=True)
        corr_top = np.dot(amat_m , amat_m .T)
       
        scales = np.linalg.norm(amat_m,  axis=0)**2
        scales_matrix = scales[:, None] * scales[None, :]

        pearson_matrix = corr_top / np.sqrt(scales_matrix)
        pearson_matrix = hollow_matrix(pearson_matrix)

        merge_inds = np.unravel_index(np.argmax(np.ravel(pearson_matrix)), pearson_matrix.shape)

        amat[merge_inds[0]] += amat[merge_inds[1]]
        amat[:, merge_inds[0]] += amat[:, merge_inds[1]]

        all_labels[merge_inds[0]] = np.hstack([
            np.squeeze(np.array(all_labels[merge_inds[0]])), 
            np.squeeze(np.array(all_labels[merge_inds[1]]))
        ]).tolist()

        all_labels = np.delete(all_labels, merge_inds[1])
        amat = np.delete(amat, merge_inds[1], axis=0)
        amat = np.delete(amat, merge_inds[1], axis=1)

    if return_labels:
        return amat, all_labels
    else:
        return amat
        



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

