"""functions related to representational similarity analysis"""
from __future__ import division, print_function, absolute_import
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform
import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed
from scipy.stats import rankdata
from functools import partial
from early_vision_toolbox.util import grouper
from itertools import izip


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

        In practice, this seems to make little difference.
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
    """

    Parameters
    ----------
    X: ndarray
    noise_level
    rng_state

    Returns
    -------

    """
    assert X.ndim == 2
    if noise_level is None:
        noise_level = 0.0
    if rng_state is None:
        rng_state = np.random.RandomState(None)
    _X = X + noise_level * rng_state.randn(*X.shape) if noise_level != 0.0 else X
    result = pdist(_X, metric='correlation')
    assert np.all(np.isfinite(result)), "I can't allow invalid values in RDM"
    return result


def rdm_similarity(ref_rdms, rdm, similarity_type='spearman', computation_method=None,
                   rdm_as_list=False):
    """

    Parameters
    ----------
    computation_method : str or None
        can be 'spearmanr' or 'rankdata+cdist'
        # these two now only apply to spearman.
        if you specify this, then you must have matching `similarity_type`.
    similarity_type : str
        can only be 'spearman' now.
    ref_rdms: ndarray
    rdm: ndarray

    Returns
    -------

    """
    if similarity_type == 'spearman' and computation_method is None:
        computation_method = 'spearmanr'

    ref_rdms = np.atleast_2d(np.asarray(ref_rdms))
    # deal with case like returning a tuple of stuff.
    rdm = np.asarray(rdm)
    assert type(rdm) == np.ndarray and type(ref_rdms) == np.ndarray
    assert ref_rdms.ndim == 2
    if not rdm_as_list:
        rdm = np.atleast_2d(rdm.ravel())  # this is 1 x N
    assert rdm.ndim == 2
    assert rdm.shape[1] == ref_rdms.shape[1]
    if computation_method == 'spearmanr':
        assert similarity_type == 'spearman'
        assert not rdm_as_list, 'only supporting one by one!'
        # if not, actually spearmanr will return a scalar instead.
        if ref_rdms.shape[0] >= 2:
            rdm_similarities = spearmanr(ref_rdms, rdm, axis=1).correlation[-1, :-1]
        else:
            # print('singular path!')
            rdm_similarities = np.atleast_1d(spearmanr(ref_rdms, rdm, axis=1).correlation)
    elif computation_method == 'rankdata+cdist':
        assert similarity_type == 'spearman'
        # do rank transform first, and then compute pearson.
        ref_rdms_ranked = np.array([rankdata(ref_rdm_this) for ref_rdm_this in ref_rdms])
        if not rdm_as_list:
            rdm_ranked = np.atleast_2d(rankdata(rdm.ravel()))
        else:
            rdm_ranked = np.array([rankdata(rdm_this) for rdm_this in rdm])
        rdm_similarities = 1 - cdist(rdm_ranked, ref_rdms_ranked, 'correlation')
    else:
        raise ValueError('unsupported computation method {}'.format(computation_method))

    # rdm_similarities will be either a 1d stuff if not rdm_as_list, or a 2d stuff if rdm_as_list.
    assert rdm_similarities.ndim == 1 or rdm_similarities.ndim == 2
    if not rdm_as_list:
        rdm_similarities = rdm_similarities.ravel()
    else:
        assert rdm_similarities.ndim == 2

    return rdm_similarities


def rdm_similarity_batch(ref_rdms, model_rdms, parallel=False, n_jobs=4, similarity_type='spearman',
                         computation_method=None, rdm_as_list=False, max_nbytes=None, verbose=5):
    """

    Parameters
    ----------
    ref_rdms
    model_rdms
    parallel
    n_jobs
    similarity_type
    computation_method
    rdm_as_list
    max_nbytes

    Returns
    -------

    """
    rdm_similarity_partial = partial(rdm_similarity, similarity_type=similarity_type,
                                     computation_method=computation_method,
                                     rdm_as_list=rdm_as_list)
    if parallel:
        # disable memmap, due to <https://github.com/numpy/numpy/issues/6750>
        # well, I found that as long as you make sure you get ndarray, not subclass of it, it's mostly fine.
        result = Parallel(n_jobs=n_jobs, verbose=verbose, max_nbytes=max_nbytes)(
            delayed(rdm_similarity_partial)(ref_rdms, rdm) for rdm in model_rdms)
    else:
        result = [rdm_similarity_partial(ref_rdms, rdm) for rdm in model_rdms]

    if not rdm_as_list:
        result = np.array(result)  # result is model rdm x ref rdm.
    else:
        result = np.concatenate(result, axis=0)

    assert result.ndim == 2
    return result


def rdm_similarity_batch_over_splits(ref_rdms_list, model_rdms, parallel=False, n_jobs=4, verbose=5):
    """ do `split_k`-way split for `iter` times, and then compute mean and std in the first and second return values,
    and return raw data in the third return value.

    :param ref_rdm:
    :param model_rdms:
    :param parallel:
    :param n_jobs:
    :return:
    """
    # TODO add full interface or rdm_similarity_batch
    raw_array = []
    n_iter = len(ref_rdms_list)
    n_total_rdm_each = None

    for i, dataset in enumerate(ref_rdms_list):
        if n_total_rdm_each is None:
            n_total_rdm_each = len(dataset)
        else:
            assert n_total_rdm_each == len(dataset)
        similarity_matrix_this = rdm_similarity_batch(dataset, model_rdms, parallel=parallel, n_jobs=n_jobs,
                                                      verbose=verbose)
        raw_array.append(similarity_matrix_this)
        if verbose != 0:
            print('done split {}/{}'.format(i + 1, n_iter))

    raw_array = np.array(raw_array)
    # print(raw_array.shape)
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


def rdm_relatedness_test(mean_ref_rdm, model_rdms, similarity_ref,
                         n=1000, similarity_type='spearman', computation_method='rankdata+cdist',
                         batch_size=50, rng_state_seed=None, parallel=True, n_jobs=4, max_nbytes=None,
                         perm_idx_list=None):  # for debug.
    """relatedness of RDM, using randomization test as in rsatoolbox

    basically, do randomization on ref rdm.

    Parameters
    ----------
    batch_size : int
        batch size for when computation_method == 'rankdata+cdist'
    mean_ref_rdm
    model_rdms
    n
    similarity_type
    computation_method
    parallel
    n_jobs

    Returns
    -------
    an array of shape (len(model_rdms),) containing the p values. Here uncorrected ones are used, for two reasons.
    1) looks like is the one used in Kregeskrote's DEMO1 in
    rsatoolbox code, used in their rsatoolbox paper (10.1371/journal.pcbi.1003553), as well as their CNN vs IT paper
    (10.1371/journal.pcbi.1003915), where neither of FDR nor FWE was mentioned.
    2) I can't make sense of their FWE correction code. there's no reference for it.
    If really some correction is needed, simply go to call p.adjust from R, and use whatever you like...
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html.

    """
    # first, compute mean ref_rdms.
    mean_ref_rdm = np.asarray(mean_ref_rdm).ravel()
    mean_ref_rdm_square = squareform(mean_ref_rdm)
    rdm_h, rdm_w = mean_ref_rdm_square.shape
    assert rdm_h == rdm_w
    model_rdms = np.asarray(model_rdms)
    assert model_rdms.ndim == 2 and model_rdms.shape[1] == mean_ref_rdm.size
    assert similarity_ref.shape == (model_rdms.shape[0],)

    # then let's get n
    rng_state = np.random.RandomState(rng_state_seed)
    mean_ref_rdm_list = []
    for i_iter in range(n):
        if perm_idx_list is None:
            perm_idx_this = rng_state.permutation(rdm_h)
        else:
            perm_idx_this = perm_idx_list[i_iter]
        assert perm_idx_this.shape == (rdm_h,)
        mean_ref_rdm_list.append(squareform(mean_ref_rdm_square[np.ix_(perm_idx_this, perm_idx_this)]))
    mean_ref_rdm_list = np.array(mean_ref_rdm_list)
    assert mean_ref_rdm_list.shape == (n, model_rdms.shape[1])

    # them compute all stuff.
    if computation_method == 'rankdata+cdist':
        # do batch stuff. notice I need to swap model rdm and mean rdm.
        similarity_matrix_null = rdm_similarity_batch(model_rdms, grouper(mean_ref_rdm_list, batch_size),
                                                      similarity_type=similarity_type,
                                                      computation_method=computation_method, parallel=parallel,
                                                      n_jobs=n_jobs, rdm_as_list=True, max_nbytes=max_nbytes)
        # after .T, we get a (n, model_rdms.shape[0]) matrix.
        assert similarity_matrix_null.shape == (n, model_rdms.shape[0])
    else:
        raise ValueError('unsupported computation method!')
    # print(similarity_matrix_null[0])
    similarity_matrix_null = np.concatenate([similarity_matrix_null, similarity_ref[np.newaxis, :]], axis=0)
    # in the original implementation, similarity_matrix_ref them selves are part of sample.
    result = 1 - np.mean(similarity_matrix_null < similarity_ref, axis=0)
    return result


def bootstrap_rdm_helper(perm_idx, ref_rdms_square, model_rdms_square, n_model_rdm, similarity_type,
                         computation_method):
    model_rdms_square_this = model_rdms_square[np.ix_(np.arange(n_model_rdm), perm_idx[1], perm_idx[1])]
    ref_rdms_square_this = ref_rdms_square[np.ix_(perm_idx[0], perm_idx[1], perm_idx[1])]

    # then flatten the images.
    mean_ref_rdms_this = np.asarray([squareform(x) for x in ref_rdms_square_this]).mean(axis=0)
    model_rdms_this = np.asarray([squareform(y) for y in model_rdms_square_this])

    # then compute the similarity!
    similarity_this_bootstrap = rdm_similarity(model_rdms_this, mean_ref_rdms_this,
                                               similarity_type=similarity_type,
                                               computation_method=computation_method)
    assert similarity_this_bootstrap.shape == (n_model_rdm,)
    return similarity_this_bootstrap


def bootstrap_rdm(ref_rdms, model_rdms, similarity_ref,
                  n=1000, similarity_type='spearman',
                  rng_state_subject_seed=None,
                  rng_state_condition_seed=None,
                  bootstrap_subject=False,
                  bootstrap_condition=True,
                  one_side=False,
                  parallel=True, n_jobs=-1, max_nbytes='1M',
                  computation_method=None,
                  perm_idx_list=None,
                  legacy=False,
                  verbose=5):
    """

    Parameters
    ----------
    legacy : bool
        whether use the legacy method exactly as in rsatoolbox. if this is true, then `one_side` must be true.
    one_side : bool
        doing one-side p test or two sides, which can be more conservative.
    ref_rdms
    model_rdms
    similarity_ref
    n
    similarity_type
    rng_state_subject_seed
    rng_state_condition_seed
    bootstrap_subject
    bootstrap_condition
    parallel
    n_jobs
    max_nbytes
    perm_idx_list: this should be a list of 1d array indices when only one of bootstrap_subject and bootstrap_subject
        is true, and a list of 2d array indices when both of them are true.

    Returns
    -------
    a (len(model_rdms), len(model_rdms)) matrix returning the raw p values for each pair of models.

    """
    if perm_idx_list is None:
        # with perm_idx_list, it's your own adventure.
        assert bootstrap_subject or bootstrap_condition, 'you must do bootstrap on something, unless you have idx list'
    # let's create reshaped rdms.
    ref_rdms_square = []
    model_rdms_square = []
    for ref_rdm in ref_rdms:
        assert ref_rdm.ndim == 1
        ref_rdms_square.append(squareform(ref_rdm))

    for model_rdm in model_rdms:  # here this model_rdms can be any iterable returing a 1d model rdm every time.
        assert model_rdm.ndim == 1
        model_rdms_square.append(squareform(model_rdm))

    ref_rdms_square = np.asarray(ref_rdms_square)
    model_rdms_square = np.asarray(model_rdms_square)

    assert ref_rdms_square.ndim == 3 and model_rdms_square.ndim == 3
    n_ref_rdm = ref_rdms_square.shape[0]
    n_model_rdm = model_rdms_square.shape[0]

    if legacy:
        assert not one_side, "legacy p-value computation only supports two side computation"
        if similarity_ref is None:  # for legacy, you can specify it to None
            similarity_ref = np.zeros((n_model_rdm,))

    assert similarity_ref.shape == (n_model_rdm,)
    rdm_h, rdm_w = ref_rdms_square.shape[1:]
    assert (rdm_h, rdm_w) == model_rdms_square.shape[1:] and rdm_h == rdm_w

    if perm_idx_list is None:
        rng_state_subject = np.random.RandomState(rng_state_subject_seed)
        rng_state_condition = np.random.RandomState(rng_state_condition_seed)
        if bootstrap_subject:
            perm_idx_list_subject_generator = (rng_state_subject.randint(n_ref_rdm, size=(n_ref_rdm,)) for _ in
                                               range(n))
        else:
            perm_idx_list_subject_generator = (np.arange(n_ref_rdm) for _ in range(n))

        if bootstrap_condition:
            perm_idx_list_condition_generator = (rng_state_condition.randint(rdm_h, size=(rdm_h,)) for _ in range(n))
        else:
            perm_idx_list_condition_generator = (np.arange(rdm_h) for _ in range(n))

        # subject then condition.
        perm_idx_list = izip(perm_idx_list_subject_generator, perm_idx_list_condition_generator)

    bootstrap_rdm_helper_partial = partial(bootstrap_rdm_helper,
                                           n_model_rdm=n_model_rdm,
                                           similarity_type=similarity_type,
                                           computation_method=computation_method)

    # then collect all.
    if parallel:
        similarity_all_bootstrap = Parallel(n_jobs=n_jobs, verbose=verbose, max_nbytes=max_nbytes)(
            delayed(bootstrap_rdm_helper_partial)(perm_idx, ref_rdms_square, model_rdms_square) for perm_idx in
            perm_idx_list)
    else:
        similarity_all_bootstrap = [bootstrap_rdm_helper_partial(perm_idx, ref_rdms_square, model_rdms_square) for
                                    perm_idx in perm_idx_list]
    similarity_all_bootstrap = np.asarray(similarity_all_bootstrap).T
    assert similarity_all_bootstrap.shape == (n_model_rdm, n)

    # then let's do the statistical analysis.
    # use ddof to be 1 to be more correct, since we are now doing statistical analysis.
    error_bars = similarity_all_bootstrap.std(axis=1, ddof=1)
    assert error_bars.shape == (n_model_rdm,)
    pairwise_p_matrix = np.empty((n_model_rdm, n_model_rdm))

    p_matrix_it = np.nditer(pairwise_p_matrix, flags=['multi_index'], op_flags=[['writeonly']])
    similarity_ref_diff = similarity_ref[:, np.newaxis] - similarity_ref[np.newaxis, :]
    assert similarity_ref_diff.shape == (n_model_rdm, n_model_rdm)
    while not p_matrix_it.finished:
        i_row, j_col = p_matrix_it.multi_index
        if i_row == j_col:
            p_matrix_it[0] = np.nan
        else:
            # get the differences.
            similarity_diff = similarity_ref_diff[i_row, j_col]
            similarity_diff_bootstrap = similarity_all_bootstrap[i_row] - similarity_all_bootstrap[j_col]
            similarity_diff_bootstrap_normed = similarity_diff_bootstrap - similarity_diff_bootstrap.mean()
            if one_side:
                p_matrix_it[0] = np.mean(similarity_diff_bootstrap_normed > similarity_diff)
            else:
                if legacy:
                    p_matrix_it[0] = 2 * min(np.mean(similarity_diff_bootstrap < 0),
                                             np.mean(similarity_diff_bootstrap > 0))
                else:
                    p_matrix_it[0] = np.mean(abs(similarity_diff_bootstrap_normed) > abs(similarity_diff))
        p_matrix_it.iternext()

    return {'bootstrap_similarity': similarity_all_bootstrap,
            'bootstrap_std': error_bars,
            'pairwise_p_matrix': pairwise_p_matrix}
