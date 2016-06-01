"""functions related to representational similarity analysis"""
from __future__ import division, print_function, absolute_import
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import time


def compute_rdm_list(X, split_k=2, split_rng_state=None, noise_level=None, noise_rng_state=None, debug=False):
    """

    :param X:
    :param split_k:
    :param rng_state:
    :return:
    """
    assert split_k >= 2, "you can't you trivial split"
    assert X.ndim == 2, "you must give a image x neuron X array"
    if split_rng_state is None:
        split_rng_state = np.random.RandomState(None)
    rng_split_sets = np.array_split(split_rng_state.permutation(X.shape[1]), split_k)
    # check the split is correct.
    assert np.array_equal(np.unique(np.concatenate(rng_split_sets)), np.arange(X.shape[1]))
    rdm_list = []
    for split_set in rng_split_sets:
        response_mean_this = X[:, split_set]
        if debug:
            print(response_mean_this.shape)
        rdm_list.append(compute_rdm(response_mean_this, noise_level, rng_state=noise_rng_state))
    final_result = np.array(rdm_list)
    assert final_result.ndim == 2 and final_result.shape[0] == split_k
    return final_result


def compute_rdm(X, noise_level=None, rng_state=None):
    if noise_level is None:
        noise_level = 0.0
    if rng_state is None:
        rng_state = np.random.RandomState(None)
    _X = X + noise_level * rng_state.randn(*X.shape) if noise_level != 0.0 else X
    result = pdist(_X, metric='correlation')
    assert np.all(np.isfinite(result)), "I can't allow invalid values in RDM"
    return result


def rdm_similarity(ref_rdms, rdm):
    """

    Parameters
    ----------
    ref_rdms: ndarray
    rdm: ndarray

    Returns
    -------

    """
    ref_rdms = np.atleast_2d(np.array(ref_rdms, copy=False))
    assert len(ref_rdms.shape) == 2
    rdm = np.atleast_2d(rdm.ravel())
    rdm_similarities = spearmanr(ref_rdms, rdm, axis=1).correlation[-1, :-1]
    return rdm_similarities


def rdm_similarity_batch(ref_rdms, model_rdms, parallel=False, n_jobs=4):
    if parallel:
        # disable memmap, due to <https://github.com/numpy/numpy/issues/6750>
        result = Parallel(n_jobs=n_jobs, verbose=5, max_nbytes=None)(
            delayed(rdm_similarity)(ref_rdms, rdm) for rdm in model_rdms)
    else:
        result = [rdm_similarity(ref_rdms, rdm) for rdm in model_rdms]
    return np.array(result)


def rdm_similarity_batch_over_splits(ref_rdm_list, model_rdms, parallel=False, n_jobs=4,
                                     split_k=2, n_iter=10,
                                     split_rng_seeds=None, noise_levels=None, noise_rng_seeds=None):
    """ do `split_k`-way split for `iter` times, and then compute mean and std in the first and second return values,
    and return raw data in the third return value.

    :param ref_rdm:
    :param model_rdms:
    :param parallel:
    :param n_jobs:
    :param split_k:
    :param n_iter:
    :param split_rng_seeds:
    :param noise_levels:
    :param noise_rng_seeds:
    :return:
    """
    raw_array = []
    n_ref_rdm_initial = len(ref_rdm_list)
    if split_rng_seeds is None:
        split_rng_seeds = [None] * n_iter
    if noise_levels is None:
        noise_levels = [None] * n_iter
    if noise_rng_seeds is None:
        noise_rng_seeds = [None] * n_iter
    assert len(split_rng_seeds) == len(noise_levels) == len(noise_rng_seeds) == n_iter

    for i in range(n_iter):
        split_rng_seeds_this = split_rng_seeds[i]
        noise_levels_this = noise_levels[i]
        noise_rng_seeds_this = noise_rng_seeds[i]

        if split_rng_seeds_this is None:
            split_rng_seeds_this = [None] * n_ref_rdm_initial
        if noise_levels_this is None:
            noise_levels_this = [None] * n_ref_rdm_initial
        if noise_rng_seeds_this is None:
            noise_rng_seeds_this = [None] * n_ref_rdm_initial
        assert len(split_rng_seeds_this) == len(noise_levels_this) == len(noise_rng_seeds_this) == n_ref_rdm_initial

        ref_rdms = []
        for j, ref_rdm in enumerate(ref_rdm_list):
            ref_rdms.extend(
                compute_rdm_list(ref_rdm, split_k, split_rng_state=np.random.RandomState(split_rng_seeds_this[j]),
                                 noise_level=noise_levels_this[j],
                                 noise_rng_state=np.random.RandomState(noise_rng_seeds_this[j])))
        assert len(ref_rdms) == n_ref_rdm_initial*split_k
        similarity_matrix_this = rdm_similarity_batch(ref_rdms, model_rdms, parallel=parallel, n_jobs=n_jobs)
        raw_array.append(similarity_matrix_this)
        print('done split {}/{}'.format(i+1, n_iter))

    raw_array = np.array(raw_array)
    print(raw_array.shape)
    assert raw_array.shape == (n_iter, len(model_rdms), split_k*n_ref_rdm_initial)

    similarity_array = np.mean(raw_array, axis=2)
    assert similarity_array.shape == (n_iter, len(model_rdms))
    mean_array = similarity_array.mean(axis=0)
    std_array = similarity_array.std(axis=0)
    assert mean_array.shape == std_array.shape == (len(model_rdms),)

    # the correlation of mean similarity w.r.t different iters.
    # should give len(model_rdms)*(len(model_rdms)-1)/2
    correlation_matrix = 1-pdist(similarity_array.T, metric='correlation')
    assert correlation_matrix.shape == (len(model_rdms)*(len(model_rdms)-1)//2, )

    return mean_array, std_array, raw_array, correlation_matrix
