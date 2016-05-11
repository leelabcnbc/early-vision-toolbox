# coding=utf-8
"""models for the classical sparse coding (in particular lasso Dictionary Learning), ICA"""

from __future__ import division, print_function, absolute_import
import numpy as np
from sklearn.decomposition import sparse_encode
from .interface import NeuronBank
from .util import make_2d_input_matrix
from spams import lasso


class LassoSparseCodingNeuronBank(NeuronBank):
    """
    the (black and white) neuron bank class for lasso sparse coding
    """

    def __init__(self, w, penalty_lambda=1.0, algorithm='lasso_cd'):
        """ initialize a Lasso Sparse Coding NeuronBank. currently, only for square images.

        Parameters
        ----------
        w : numpy.ndarray
            dictionary. num neuron x num features.
        penalty_lambda : float
            penalty term. this penalty term is same as lambda in ``mexLasso`` of
            SPAMS (http://spams-devel.gforge.inria.fr/doc/html/doc_spams005.html#sec15)
            in practice, the solution is provided by scikit-learn, which has the SAME definition of lambda.

        """
        super(LassoSparseCodingNeuronBank, self).__init__()
        w = np.atleast_2d(w)
        assert len(w.shape) == 2
        n_filter, filtersizesq = w.shape
        filtersize = np.int(np.sqrt(filtersizesq))
        assert filtersize ** 2 == filtersizesq, "filter must be originally square!"
        if w.dtype is not np.float64:
            w = w.astype(np.float64)  # use float64
        self.w = w
        self.__filtersize = filtersize
        assert np.isscalar(penalty_lambda)
        self._lambda = penalty_lambda
        self.last_cost = np.nan
        self.algorithm = algorithm

    @property
    def n_neuron(self):
        """ number of neurons

        Returns
        -------

        """
        return self.w.shape[0]

    @property
    def rf_size(self):
        """ size of RF

        Returns
        -------

        """
        return self.__filtersize, self.__filtersize

    def predict(self, imgs, neuron_idx=None, penalty_lambda=None, algorithm=None):
        """ get neuron response to images

        Parameters
        ----------
        imgs

        Returns
        -------

        """
        imgs_array = make_2d_input_matrix(imgs)
        if neuron_idx is None:
            dict_to_use = self.w
        else:
            dict_to_use = self.w[neuron_idx:(neuron_idx + 1), :]

        if penalty_lambda is None:
            _lambda = self._lambda
        else:
            _lambda = penalty_lambda
        assert np.isscalar(_lambda)

        if algorithm is None:
            _algorithm = self.algorithm
        else:
            _algorithm = algorithm


        # let's call sparse encoder to do it!
        # no scaling at all!
        # having /nsample in objective function is exactly the same as sovling each problem separately.
        # the underlying function called is elastic net, and that function fits each column of y separately.
        # each column of y is each stimulus. This is because when passing imgs_array and dict_to_use to Elastic Net,
        # they are transposed. That is, y = imgs_array.T
        #
        # in the code there's also a subtle detail, where alpha is divided by number of pixels in each stimulus.
        # I haven't figured that out, but seems that's simply a detail for using ElasticNet to do this.
        if _algorithm in ['lasso_lars', 'lasso_cd']:
            response = sparse_encode(imgs_array, dict_to_use, alpha=_lambda, algorithm=_algorithm, max_iter=10000)
        else:
            assert _algorithm == 'spams'
            #print(imgs_array.dtype, dict_to_use.dtype, _lambda.shape)
            response = lasso(np.asfortranarray(imgs_array.T), D=np.asfortranarray(dict_to_use.T), lambda1=_lambda,
                             mode=2)
            response = response.T.toarray()  # because lasso returns sparse matrix...
        # this can be used for debugging, for comparison with SPAMS.
        # notice here I give per sample cost.
        self.last_cost = 0.5 * np.sum((imgs_array - np.dot(response, dict_to_use)) ** 2) + _lambda * np.abs(response).sum()
        self.last_cost /= imgs_array.shape[0]

        return response
