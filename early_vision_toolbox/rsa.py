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
    assert X.ndim == 2, "you must give a image x neuron X array"
    if split_rng_state is None:
        split_rng_state = np.random
    rng_split_sets = np.array_split(split_rng_state.permutation(X.shape[1]), split_k)
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
        rng_state = np.random  # use default generator
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


def rdm_similarity_batch(ref_rdms, model_rdms, parallel=False, n_jobs=4, timing=True):
    t = time.time()
    if parallel:
        # disable memmap, due to <https://github.com/numpy/numpy/issues/6750>
        result = Parallel(n_jobs=n_jobs, verbose=5, max_nbytes=None)(
            delayed(rdm_similarity)(ref_rdms, rdm) for rdm in model_rdms)
    else:
        result = [rdm_similarity(ref_rdms, rdm) for rdm in model_rdms]
    elapsed = time.time() - t
    if timing:
        print("job done in {} seconds".format(elapsed))
    return np.array(result)
