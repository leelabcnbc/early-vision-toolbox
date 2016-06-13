from __future__ import absolute_import, division, print_function
import unittest
from early_vision_toolbox.rsa import compute_rdm_bounds
import numpy as np
import h5py
from scipy.spatial.distance import squareform

class MyTestCase(unittest.TestCase):
    def test_rsa_bounds_spearman(self):
        with h5py.File('rsa_ref/rsa_ref.hdf5', 'r') as f:
            rdm_stack_all = f['rsa_bounds/rdm_stack_all'][...] # don't need transpose, as it's symmetric.
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


if __name__ == '__main__':
    unittest.main()
