from __future__ import division, print_function, absolute_import
import unittest
import h5py
import numpy as np
from early_vision_toolbox.preprocessing import whiten_olsh_lee_inner


class MyTestCase(unittest.TestCase):
    def test_whiten_olsh_lee_inner(self):
        # first, load files.
        f = h5py.File('preprocessing_ref/one_over_f_whitening_ref.hdf5', 'r')
        grp = f['testcases_matlab']
        old_images = grp['original_images'][:].transpose((0, 2, 1))
        new_images = grp['new_images'][:].transpose((0, 2, 1))
        f.close()

        for old_im, new_im in zip(old_images, new_images):
            new_im_test = whiten_olsh_lee_inner(old_im)
            self.assertTrue(np.allclose(new_im, new_im_test))



if __name__ == '__main__':
    unittest.main()
