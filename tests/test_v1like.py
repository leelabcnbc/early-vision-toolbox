from __future__ import absolute_import, division, print_function
import unittest
from early_vision_toolbox.v1like import v1like
from early_vision_toolbox.v1like.legacy import v1s_misc
from early_vision_toolbox.util import normalize_vector_inplace
import numpy as np
import random

import time
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[{}]'.format(self.name), end='')
        print('Elapsed: {}'.format(time.time() - self.tstart))


tol = 1e-6
# from time import time



class MyTestCase(unittest.TestCase):
    def get_new_image(self, ndim=2):
        shape_this = np.random.randint(50, 200, (ndim,), dtype=np.uint16)
        image = np.random.randn(*shape_this).astype(np.float32)
        return image


    @unittest.skip('this is useless, because imresize from Pillow is different from resize in skimage')
    def test_resize(self):
        shape_this = np.random.randint(50, 200, (2,), dtype=np.uint16)
        image = np.random.randint(28, 228, size=(shape_this[0], shape_this[1], 3), dtype=np.uint8)
        image_processed_1 = v1like._preproc_resize(image, 150, True)
        image_processed_2 = v1like._preproc_resize(image, 150, False)
        print(image.max())
        print(image_processed_1.dtype, image_processed_1.max())
        print(image_processed_2.dtype, image_processed_2.max())
        print(abs(image_processed_1 - image_processed_2).max())


    def test_lowpass(self):
        for _ in range(1000):
            image = self.get_new_image()
            image_processed_1 = v1like._preproc_lowpass(image, 3, True)
            image_processed_2 = v1like._preproc_lowpass(image, 3, False)
            self.assertTrue(np.allclose(image_processed_1, image_processed_2, atol=tol))
            self.assertEqual(image_processed_1.shape, image_processed_2.shape)


    def test_local_normalization(self):
        for _ in range(10):
            image = self.get_new_image()
            kshape_this = np.random.choice([3, 5, 7, 9, 11])
            threshold_this = np.random.rand() * 10.0
            params = {'kshape': (kshape_this, kshape_this), 'threshold': threshold_this}
            # t1 = time()

            # t2 = time()
            # print('old take {}'.format(t2-t1))
            # t1 = time()
            image_processed_2 = v1like._normin(image, params, False)
            image_processed_1 = v1like._normin(image, params, True)
            # t2 = time()
            # print('new take {}'.format(t2 - t1))
            # print(shape_this)
            # print(image_processed_1.shape, image_processed_2.shape)
            self.assertTrue(np.allclose(image_processed_1, image_processed_2, atol=tol))
            self.assertEqual(image_processed_1.shape, image_processed_2.shape)


    def test_rdim(self):
        for _ in range(10):
            for lsum_ksize in [1,3,5,7,9]:
                image = self.get_new_image(3)
                outshape = np.random.randint(10, 20, (2,), dtype=np.uint16)
                image_processed_1 = v1like._dimr(image, lsum_ksize, outshape)
                image_processed_2 = v1like._dimr(image, lsum_ksize, outshape, False)
                self.assertTrue(np.allclose(image_processed_1, image_processed_2, atol=tol))
                self.assertEqual(image_processed_1.shape, image_processed_2.shape)

    def test_filter(self):
        norients = 16
        orients = [o * np.pi / norients for o in range(norients)]
        divfreqs = [2, 3, 4, 6, 11, 18]
        freqs = [1. / n for n in divfreqs]
        phases = [0]

        for kshape in range(12, 16):
            filter_params = {
                # kernel shape of the gabors
                'kshape': (kshape, kshape),
                # list of orientations
                'orients': orients,
                # list of frequencies
                'freqs': freqs,
                # list of phases
                'phases': phases,
                # threshold (variance explained) for the separable convolution
                # should be set to 1 or bigger when debugging.
                'sep_threshold': 1.1,
                'max_component': 100000,  # to reduce error,
                'fix_bug': True
            }

            filt_l, filt_l_raw = v1s_misc._get_gabor_filters(filter_params)
            for _ in range(5):
                image = self.get_new_image()
                normalize_vector_inplace(image)
                #with Timer('legacy'):
                conv_result_1 = v1like._filter(image, filt_l, True)
                #with Timer('new'):
                conv_result_2 = v1like._filter(image, filt_l_raw, False)
                self.assertTrue(np.allclose(conv_result_1, conv_result_2, atol=tol))
                self.assertEqual(conv_result_1.shape, conv_result_2.shape)



if __name__ == '__main__':
    unittest.main()
