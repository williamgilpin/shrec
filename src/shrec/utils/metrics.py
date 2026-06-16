import sklearn.metrics
from sklearn.decomposition import PCA
import numpy as np

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
    # nbins = np.sqrt(n)
    nbins = int(round(1 + np.log2(n)))

    data = np.sort(data)[::-1]

    hist, bins = np.histogram(data, nbins)
    bins = bins[:-1] + (bins[1] - bins[0]) / 2  # center bins
    hist = hist.astype(float)
    hist /= np.sum(hist)  # normalized

    bin_index, cross_var_highest = nbins // 2, -1
    for i in np.arange(1, nbins - 1):
        pleft, pright = np.sum(hist[:i]), np.sum(hist[i:])

        mean_left = np.sum(hist[:i] * bins[:i]) / pleft
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
    thresh = np.percentile(np.ravel(np.abs(a)), 100 * sparsity, method="higher")
    a[np.abs(a) <= thresh] = 0  # sparsify
    if weighted:
        pass
    else:
        a[np.abs(a) > thresh] = 1
    return a


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
    # print(wv.shape)
    sel_inds = np.cumsum(pca.explained_variance_ratio_) < cutoff
    # print(sel_inds.shape)
    pc = pca.components_
    pc_truncated = pc[sel_inds]

    X_recon = np.dot(wv[:, sel_inds], pc[sel_inds])

    scores = np.linalg.norm(X - X_recon, axis=-1)
    return scores

# Baselines


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
