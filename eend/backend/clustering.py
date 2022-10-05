import numpy as np
from scipy.special import softmax


def twoGMMcalib_lin(s, niters=10, var_eps=1e-12):
    """
    Train two-Gaussian GMM with shared variance for calibration of scores 's'
    Returns threshold for original scores 's' that "separates" the two gaussians
    and array of linearly callibrated log odds ratio scores.
    """
    weights = np.array([0.5, 0.5])
    means = np.mean(s) + np.std(s) * np.array([-1, 1])
    var = np.var(s)
    for _ in range(niters):
        lls = np.log(weights) - 0.5 * np.log(var) - 0.5 * (s[:, np.newaxis] - means) ** 2 / var
        gammas = softmax(lls, axis=1)
        cnts = np.sum(gammas, axis=0)
        weights = cnts / cnts.sum()
        means = s.dot(gammas) / cnts
        var = ((s ** 2).dot(gammas) / cnts - means ** 2).dot(weights)
        if abs(var) < var_eps:  # avoid division by zero
            break
    threshold = -0.5 * (np.log(weights ** 2 / var) - means ** 2 / var).dot([1, -1]) / (means / var).dot([1, -1])
    return threshold


def AHC(sim_mx, threshold=0):
    """ Performs UPGMA variant (wikipedia.org/wiki/UPGMA) of Agglomerative
    Hierarchical Clustering using the input pairwise similarity matrix.
    Input:
        sim_mx    - NxN pairwise similarity matrix
        threshold - threshold for stopping the clustering algorithm
                    (see function twoGMMcalib_lin for its estimation)
    Output:
        cluster labels stored in an array of length N containing (integers in
        the range from 0 to C-1, where C is the number of dicovered clusters)
    """
    dist = -sim_mx;
    dist[np.diag_indices_from(dist)] = np.inf
    clsts = [[i] for i in range(len(dist))]
    while True:
        mi, mj = np.sort(np.unravel_index(dist.argmin(), dist.shape))
        if dist[mi, mj] > -threshold:
            break
        dist[:, mi] = dist[mi, :] = (dist[mi, :] * len(clsts[mi]) + dist[mj, :] * len(clsts[mj])) / (
                    len(clsts[mi]) + len(clsts[mj]))
        dist[:, mj] = dist[mj, :] = np.inf
        clsts[mi].extend(clsts[mj])
        clsts[mj] = None
    labs = np.empty(len(dist), dtype=int)
    for i, c in enumerate([e for e in clsts if e]):
        labs[c] = i
    return labs

def l2_norm(vec_or_matrix):
    """ L2 normalization of vector array.

    Args:
        vec_or_matrix (np.array): one vector or array of vectors

    Returns:
        np.array: normalized vector or array of normalized vectors
    """
    if len(vec_or_matrix.shape) == 1:
        # linear vector
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    elif len(vec_or_matrix.shape) == 2:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]
    else:
        raise ValueError('Wrong number of dimensions, 1 or 2 is supported, not %i.' % len(vec_or_matrix.shape))


def cos_similarity(x):
    """Compute cosine similarity matrix in CPU & memory sensitive way

    Args:
        x (np.ndarray): embeddings, 2D array, embeddings are in rows

    Returns:
        np.ndarray: cosine similarity matrix

    """

    assert x.ndim == 2, f'x has {x.ndim} dimensions, it must be matrix'
    x = x / (np.sqrt(np.sum(np.square(x), axis=1, keepdims=True)) + 1.0e-32)
    assert np.allclose(np.ones_like(x[:, 0]), np.sum(np.square(x), axis=1))
    max_n_elm = 200000000
    step = max(max_n_elm // (x.shape[0] * x.shape[0]), 1)
    retval = np.zeros(shape=(x.shape[0], x.shape[0]), dtype=np.float64)
    x0 = np.expand_dims(x, 0)
    x1 = np.expand_dims(x, 1)
    for i in range(0, x.shape[1], step):
        product = x0[:, :, i:i + step] * x1[:, :, i:i + step]
        retval += np.sum(product, axis=2, keepdims=False)
    assert np.all(retval >= -1.0001), retval
    assert np.all(retval <= 1.0001), retval
    return retval