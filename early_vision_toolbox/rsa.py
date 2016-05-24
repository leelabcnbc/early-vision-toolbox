"""functions related to representational similarity analysis"""
from __future__ import division, print_function, absolute_import
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import time


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
