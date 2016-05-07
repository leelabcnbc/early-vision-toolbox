from __future__ import division, print_function, absolute_import
import unittest
import h5py
import numpy as np
from early_vision_toolbox import sparse_coding


class MyTestCase(unittest.TestCase):
    def test_lasso(self):
        # first, load the file
        f = h5py.File('sparse_coding_ref/sparse_coding_ref.hdf5', 'r+')
        # get the filter
        grp = f['spams/mexLasso/mode2']
        w = grp['W'][:]
        # print(w.shape)
        # get image
        images = grp['images'][:]
        # use this method for maximum accuracy.  'lasso_cd' is much faster but a little less accurate.
        model = sparse_coding.LassoSparseCodingNeuronBank(w, algorithm='lasso_lars')
        lambda_list = grp.attrs['lambda_list']
        cost_list = grp.attrs['cost_list']

        response_ref_list = []
        for idx in range(cost_list.size):
            response_this_ref = grp['response/' + str(idx + 1)][:]
            assert np.all(np.logical_not(np.isnan(response_this_ref)))
            response_ref_list.append(response_this_ref)
        f.close()

        for idx, penalty_lambda in enumerate(lambda_list):
            # print(penalty_lambda)
            response_this = model.predict(images, penalty_lambda=penalty_lambda)
            cost_this = model.last_cost * images.shape[0]
            cost_this_ref = cost_list[idx]
            response_this_ref = response_ref_list[idx]
            cost_diff = np.abs(cost_this - cost_this_ref)
            resp_diff = np.abs(response_this_ref - response_this).max()
            # print('cost diff {} - {} = {}'.format(cost_this, cost_this_ref, cost_diff))
            # print('max response diff {}'.format(resp_diff))
            self.assertTrue(cost_diff < 1e-6)
            self.assertTrue(resp_diff < 1e-3)


if __name__ == '__main__':
    unittest.main()
