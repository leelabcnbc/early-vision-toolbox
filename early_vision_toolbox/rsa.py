"""functions related to representational similarity analysis"""
from __future__ import division, print_function, absolute_import
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import time
from scipy.stats import rankdata

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


def compute_rdm_list_batch(ref_feature_matrix_list,
                           split_k=2, n_iter=10,
                           split_rng_seeds=None, noise_levels=None, noise_rng_seeds=None):
    rdm_list_list = []
    n_ref_rdm_initial = len(ref_feature_matrix_list)
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
        for j, ref_feature_matrix in enumerate(ref_feature_matrix_list):
            # here it's extending a list with numpy array, and it will work as expected...
            ref_rdms.extend(
                compute_rdm_list(ref_feature_matrix, split_k,
                                 split_rng_state=np.random.RandomState(split_rng_seeds_this[j]),
                                 noise_level=noise_levels_this[j],
                                 noise_rng_state=np.random.RandomState(noise_rng_seeds_this[j])))
        assert len(ref_rdms) == n_ref_rdm_initial * split_k
        rdm_list_list.append(ref_rdms)
    return np.asarray(rdm_list_list)  # for compact.


def compute_rdm_bounds(rdm_list, similarity_type='spearman', legacy=True):
    """computes the estimated lower and upper bounds of the similarity between the ideal model and the given data..
    this is a remake of ``ceilingAvgRDMcorr`` in rsatoolbox in MATLAB.

    Parameters
    ----------
    legacy : bool
        whether behave exactly the same as in rsatoolbox. maybe this is the correct behavior.
        so basically, in lower bound computation, at each time, we compute the best rdm for all but one RDMs,
        and we hope this RDM to under fit. (I don't know whether one rdm's difference will turn overfit to underfit).
    rdm_list
    similarity_type : str
        type of similarity. only spearman is supported currently.

    Returns
    -------

    """
    rdm_list = np.asarray(rdm_list)
    assert rdm_list.ndim == 2
    n_rdm = rdm_list.shape[0]
    assert n_rdm >= 3, 'at least 3 RDMs to compute bounds (with 2, cross validation cannot be done)'
    if similarity_type == 'spearman':
        # transform everything to rank
        rdm_list_rank = np.empty_like(rdm_list)
        for idx, rdm in enumerate(rdm_list):
            rdm_list_rank[idx] = rankdata(rdm)
        if legacy:  # use rank transformed data, even for lower bound.
            rdm_list = rdm_list_rank
        best_rdm = rdm_list_rank.mean(axis=0)
        upper_bound = rdm_similarity(rdm_list, best_rdm).mean()
    else:
        raise ValueError('supported similarity type!')

    # compute lower bound. cross validation.
    # maybe it's good to use np.bool_ rather than np.bool
    # check <https://github.com/numba/numba/issues/1311>
    all_true_vector = np.ones((n_rdm,), dtype=np.bool_)
    similarity_list_for_lowerbound = []
    for i_rdm in range(n_rdm):
        selection_vector_this = all_true_vector.copy()
        selection_vector_this[i_rdm] = False
        similarity_list_for_lowerbound.append(spearmanr(rdm_list[selection_vector_this].mean(axis=0),
                                                        rdm_list[i_rdm]).correlation)
    lower_bound = np.array(similarity_list_for_lowerbound).mean()
    return lower_bound, upper_bound


def compute_rdm_bounds_batch(ref_rdms_list, similarity_type='spearman'):
    """

    Parameters
    ----------
    ref_feature_matrix_list
    split_k
    n_iter
    split_rng_seeds
    noise_levels
    noise_rng_seeds
    similarity_type

    Returns
    -------

    """
    lower_bound_array = []
    upper_bound_array = []
    n_iter = len(ref_rdms_list)

    for i, ref_rdms in enumerate(ref_rdms_list):
        lower_bound, upper_bound = compute_rdm_bounds(ref_rdms, similarity_type)
        lower_bound_array.append(lower_bound)
        upper_bound_array.append(upper_bound)

    lower_bound_array = np.array(lower_bound_array)
    upper_bound_array = np.array(upper_bound_array)
    assert lower_bound_array.shape == upper_bound_array.shape == (n_iter,)

    return lower_bound_array, upper_bound_array


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
    assert len(ref_rdms.shape) == 2  # this is K x N
    # if not, actually spearmanr will return a scalar instead.
    assert ref_rdms.shape[0] >= 2, 'at least two ref rdms or more'
    rdm = np.atleast_2d(rdm.ravel())  # this is 1 x N
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


def rdm_similarity_batch_over_splits(ref_rdms_list, model_rdms, parallel=False, n_jobs=4):
    """ do `split_k`-way split for `iter` times, and then compute mean and std in the first and second return values,
    and return raw data in the third return value.

    :param ref_rdm:
    :param model_rdms:
    :param parallel:
    :param n_jobs:
    :return:
    """
    raw_array = []
    n_iter = len(ref_rdms_list)
    n_total_rdm_each = None

    for i, dataset in enumerate(ref_rdms_list):
        if n_total_rdm_each is None:
            n_total_rdm_each = len(dataset)
        else:
            assert n_total_rdm_each == len(dataset)
        similarity_matrix_this = rdm_similarity_batch(dataset, model_rdms, parallel=parallel, n_jobs=n_jobs)
        raw_array.append(similarity_matrix_this)
        print('done split {}/{}'.format(i + 1, n_iter))

    raw_array = np.array(raw_array)
    print(raw_array.shape)
    assert raw_array.shape == (n_iter, len(model_rdms), n_total_rdm_each)

    similarity_array = np.mean(raw_array, axis=2)
    assert similarity_array.shape == (n_iter, len(model_rdms))
    mean_array = similarity_array.mean(axis=0)
    std_array = similarity_array.std(axis=0)
    assert mean_array.shape == std_array.shape == (len(model_rdms),)

    # the correlation of mean similarity w.r.t different iters.
    # should give len(model_rdms)*(len(model_rdms)-1)/2
    correlation_matrix = 1 - pdist(similarity_array.T, metric='correlation')
    assert correlation_matrix.shape == (len(model_rdms) * (len(model_rdms) - 1) // 2,)

    return mean_array, std_array, raw_array, correlation_matrix
