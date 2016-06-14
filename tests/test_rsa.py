from __future__ import absolute_import, division, print_function
import unittest
from early_vision_toolbox.rsa import compute_rdm_bounds, rdm_similarity, rdm_similarity_batch
import numpy as np
import h5py
from scipy.spatial.distance import squareform, pdist
import time
from early_vision_toolbox.util import grouper


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[{}]'.format(self.name), end='')
        print('Elapsed: {}'.format(time.time() - self.tstart))


class MyTestCase(unittest.TestCase):
    def test_rsa_bounds_spearman(self):
        with h5py.File('rsa_ref/rsa_ref.hdf5', 'r') as f:
            rdm_stack_all = f['rsa_bounds/rdm_stack_all'][...]  # don't need transpose, as it's symmetric.
            result_array = f['rsa_bounds/result_array'][...].T
        for idx, rdm_stack in enumerate(rdm_stack_all):
            rdm_vector_list = []
            for rdm in rdm_stack:
                rdm_vector_list.append(squareform(rdm))
            rdm_vector_list = np.array(rdm_vector_list)
            lower_bound, upper_bound = compute_rdm_bounds(rdm_vector_list, similarity_type='spearman', legacy=True)
            lower_bound_ref = result_array[idx, 1]
            upper_bound_ref = result_array[idx, 0]
            self.assertTrue(np.allclose(lower_bound, lower_bound_ref))
            self.assertTrue(np.allclose(upper_bound, upper_bound_ref))

    def test_rsa_similarity_two_methods(self):
        """ test performance of two ways to compute spearman similarity

        Returns
        -------

        """
        # first create all rdms.
        rng_state = np.random.RandomState(0)

        for i_test in range(2):
            # 10 rdm matrices. here size is in the same order as those in Tang's data.
            ref_rdms = np.array([pdist(rng_state.randn(2000, 100), 'correlation') for _ in range(5)])
            assert ref_rdms.shape == (5, 2000 * 1999 / 2)
            # then one rdm
            model_rdm_list = np.array([pdist(rng_state.randn(2000, 100), 'correlation') for _ in range(100)])
            assert model_rdm_list.shape == (100, 2000 * 1999 / 2)
            with Timer('spearmanr'):
                a = rdm_similarity_batch(ref_rdms, model_rdm_list, computation_method='spearmanr', parallel=True,
                                         n_jobs=-1)

            with Timer('rankdata+cdist'):
                # use grouper_no_padding to feed data in batch...
                # get the best of both worlds... small memory print (from iterator) AND more efficient computation!
                b = rdm_similarity_batch(ref_rdms, grouper(model_rdm_list, 50),
                                         computation_method='rankdata+cdist', parallel=True,
                                         n_jobs=-1, rdm_as_list=True, max_nbytes='1M')
            # print(a.shape, b.shape)
            self.assertTrue(a.shape == b.shape)
            self.assertTrue(np.allclose(a, b))
            print(i_test)


if __name__ == '__main__':
    unittest.main()
