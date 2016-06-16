from __future__ import absolute_import, division, print_function
import unittest
from early_vision_toolbox.rsa import (compute_rdm_bounds, rdm_similarity_batch,
                                      rdm_relatedness_test, bootstrap_rdm, compute_rdm)
import numpy as np
import h5py
from scipy.spatial.distance import squareform, pdist
import time
from early_vision_toolbox.util import grouper
from scipy.io import loadmat


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

    def test_rsa_relatedness(self):
        ref_mat = loadmat('rsa_ref/debug_rsa_relatedness.mat')
        rdm_stack_all = ref_mat['rdm_stack_all']
        cand_rdm_stack_all = ref_mat['cand_rdm_stack_all']
        index_matrix_array = ref_mat['index_matrix_array']
        p_value_array = ref_mat['p_value_array']
        # print(rdm_stack_all.shape, cand_rdm_stack_all.shape, index_matrix_array.shape, p_value_array.shape)

        for i_case in range(p_value_array.shape[-1]):
            ref_rdms = rdm_stack_all[:, :, :, i_case]

            if i_case % 2 != 0:
                ref_rdms = ref_rdms[:, :, :1]  # check singular case.

            ref_rdms = np.array([squareform(ref_rdms[:, :, x]) for x in range(ref_rdms.shape[2])])
            cand_rdms = cand_rdm_stack_all[:, :, :, i_case]
            cand_rdms = np.array([squareform(cand_rdms[:, :, x]) for x in range(cand_rdms.shape[2])])
            # compute similarity.
            similarity_matrix_ref = rdm_similarity_batch(ref_rdms, cand_rdms, computation_method='spearmanr').mean(
                axis=1)
            p_val_this = rdm_relatedness_test(mean_ref_rdm=ref_rdms.mean(axis=0), model_rdms=cand_rdms,
                                              similarity_ref=similarity_matrix_ref,
                                              n=100, perm_idx_list=index_matrix_array[:, :, i_case].T - 1)
            p_val_ref = p_value_array[:, i_case]
            assert p_val_this.shape == p_val_ref.shape
            # print(p_val_this - p_val_ref)
            # print(abs(p_val_this - p_val_ref).max())
            self.assertTrue(np.allclose(p_val_this, p_val_ref))

    def test_rsa_bootstrap(self):
        """ test bootstrapping of rdm

        Returns
        -------

        """
        ref_mat = loadmat('rsa_ref/debug_rsa_bootstrap.mat')
        rdm_stack_all = ref_mat['rdm_stack_all']
        cand_rdm_stack_all = ref_mat['cand_rdm_stack_all']
        index_matrix_array_condition = ref_mat['index_matrix_array_condition']
        index_matrix_array_subject = ref_mat['index_matrix_array_subject']
        p_value_array = ref_mat['p_value_array']
        bootstrap_e_array = ref_mat['bootstrap_e_array']
        # print(rdm_stack_all.shape, cand_rdm_stack_all.shape, index_matrix_array.shape, p_value_array.shape)
        similarity_array = ref_mat['similarity_array']
        for i_case in range(p_value_array.shape[-1]):
            ref_rdms = rdm_stack_all[:, :, :, i_case]

            index_matrix_array_condition_this = index_matrix_array_condition[:, :, i_case]
            index_matrix_array_subject_this = index_matrix_array_subject[:, :, i_case]

            if i_case % 2 != 0:
                ref_rdms = ref_rdms[:, :, :1]  # check singular case.
                index_matrix_array_subject_this = index_matrix_array_subject_this[:, :1]

            # construct perm_idx_list
            perm_idx_list = zip(index_matrix_array_subject_this - 1, index_matrix_array_condition_this - 1)
            assert len(perm_idx_list) == 250

            ref_rdms = np.array([squareform(ref_rdms[:, :, x]) for x in range(ref_rdms.shape[2])])
            cand_rdms = cand_rdm_stack_all[:, :, :, i_case]
            cand_rdms = np.array([squareform(cand_rdms[:, :, x]) for x in range(cand_rdms.shape[2])])
            # compute similarity.
            result_this = bootstrap_rdm(ref_rdms=ref_rdms, model_rdms=cand_rdms, similarity_ref=None,
                                        n=250, legacy=True, perm_idx_list=perm_idx_list)
            p_val_ref = p_value_array[:, :, i_case]
            p_val_this = result_this['pairwise_p_matrix']
            assert p_val_this.shape == p_val_ref.shape and p_val_ref.dtype == p_val_this.dtype
            # print(p_val_this - p_val_ref)
            self.assertTrue(np.allclose(p_val_this, p_val_ref, equal_nan=True))
            # print(p_val_this)
            # print(abs(p_val_this[np.isfinite(p_val_this)] - p_val_ref[np.isfinite(p_val_ref)]).max())
            bootstrap_e_ref = bootstrap_e_array[:, i_case]
            bootstrap_e_this = result_this['bootstrap_std']
            assert bootstrap_e_ref.shape == bootstrap_e_this.shape and bootstrap_e_ref.dtype == bootstrap_e_this.dtype
            self.assertTrue(np.allclose(bootstrap_e_ref, bootstrap_e_this))
            # print(bootstrap_e_ref - bootstrap_e_this)
            similarity_this = result_this['bootstrap_similarity']
            similarity_ref = similarity_array[:, :, i_case]
            # print(abs(raw_similarity_ref - raw_similarity_this).max())
            assert similarity_this.shape == similarity_ref.shape and similarity_this.dtype == similarity_ref.dtype
            self.assertTrue(np.allclose(similarity_ref, similarity_this))


if __name__ == '__main__':
    unittest.main(failfast=True)