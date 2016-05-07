from __future__ import division, print_function, absolute_import
import unittest
import h5py
import numpy as np
from early_vision_toolbox import tuning


class MyTestCase(unittest.TestCase):
    def test_find_optimal_paras_rf(self):
        # first, load the file
        f = h5py.File('tuning_ref_results/NIS_results.hdf5', 'r+')
        # get the filter
        grp = f['ICA/tuning']
        wica = grp['Wica'][:]  # should be
        # get the reference results
        ica_optx = grp.attrs['ica_optx']
        ica_opty = grp.attrs['ica_opty']
        ica_optfreq = grp.attrs['ica_optfreq']
        ica_optor = grp.attrs['ica_optor']
        ica_optphase = grp.attrs['ica_optphase']

        # now shuffle this Wica. 1024x256
        wica = wica.T
        wica = wica.reshape(256, 32, 32)
        wica = np.transpose(wica, (0, 2, 1))
        wica = wica.reshape(256, 1024)

        # get result
        result = tuning.find_optimal_paras_rf(w=wica, legacy=True)

        # print result
        self.assertTrue(np.allclose(result['optx'], ica_optx))
        self.assertTrue(np.allclose(result['opty'], ica_opty))
        self.assertTrue(np.array_equal(result['optfreq'], ica_optfreq))
        self.assertTrue(np.array_equal(result['optor'], ica_optor))
        self.assertTrue(np.array_equal(result['optphase'], ica_optphase))
        f.close()


if __name__ == '__main__':
    unittest.main()
