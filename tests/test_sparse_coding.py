from __future__ import division, print_function, absolute_import
import unittest
import h5py
import numpy as np
from early_vision_toolbox import sparse_coding


class MyTestCase(unittest.TestCase):
    def test_lasso(self):
        # first, load the file
        f = h5py.File('sparse_coding_ref/sparse_coding_ref.hdf5', 'r')
        # get the filter
        grp = f['spams/mexLasso/mode2']
        w = grp['W'][:]
        # print(w.shape)
        # get image
        images = grp['images'][:]
        # use this method for maximum accuracy.  'lasso_cd' is much faster but a little less accurate.
        model = sparse_coding.LassoSparseCodingNeuronBank(w, algorithm='spams')
        lambda_list = grp.attrs['lambda_list']
        cost_list = grp.attrs['cost_list']

        response_ref_list = []
        for idx in range(cost_list.size):
            response_this_ref = grp['response/' + str(idx + 1)][:]
            self.assertTrue(np.all( np.isfinite(response_this_ref)))
            response_ref_list.append(response_this_ref)
        f.close()

        for idx, penalty_lambda in enumerate(lambda_list):
            # print(penalty_lambda)
            response_this = model.predict(images, penalty_lambda=penalty_lambda)
            cost_this = model.last_cost * images.shape[0]
            cost_this_ref = cost_list[idx]
            response_this_ref = response_ref_list[idx]
            self.assertTrue(np.allclose(response_this_ref, response_this))
            self.assertTrue(np.allclose(cost_this_ref, cost_this))


if __name__ == '__main__':
    unittest.main()
