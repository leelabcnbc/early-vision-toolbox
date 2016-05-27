from __future__ import absolute_import, division, print_function
import unittest
from early_vision_toolbox.v1like import v1like
from early_vision_toolbox.v1like.legacy import v1s_misc, v1s_funcs
from early_vision_toolbox.util import normalize_vector_inplace
import numpy as np
from scipy.io import loadmat
from skimage.io import imread
import time

debug = True


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if debug:
            if self.name:
                print('[{}]'.format(self.name), end='')
            print('Elapsed: {}'.format(time.time() - self.tstart))


tol = 1e-5


# from time import time



class MyTestCase(unittest.TestCase):
    # @unittest.skip('for now')
    def test_reference_legacy(self):
        """compare with reference result of original implementation"""
        image_list = ['./v1like_ref/sample_{}.png'.format(i) for i in range(10)]
        reference_result = loadmat('./v1like_ref/reference_v1like_result.mat')['feature_matrix']
        #old_images = loadmat('./v1like_ref/reference_v1like_result.mat')['images_after_resize']
        # now let's get them.
        # use default parameters.
        # get the fitlers.
        pars = v1like.default_pars()
        filt_l, filt_l_raw = v1s_misc._get_gabor_filters(pars['representation']['filter'])
        result = []
        with Timer('legacy version'):
            for idx, imagename in enumerate(image_list):
                im = imread(imagename)
                result.append(v1like._part_generate_repr(im, pars['steps'],
                                                         pars['representation'], pars['featsel'], filt_l, legacy=True,
                                                         debug=True))
                print("finish image {}".format(idx + 1))

        result_legacy = np.array(result)
        self.assertEqual(reference_result.dtype, result_legacy.dtype)
        self.assertEqual(reference_result.shape, result_legacy.shape)
        if debug:
            print(abs(reference_result[:, :] - result_legacy[:, :]).max())
        self.assertTrue(np.allclose(reference_result, result_legacy, atol=tol))

    def test_reference_legacy_faster(self):
        """compare with reference result of original implementation"""
        image_list = ['./v1like_ref/sample_{}.png'.format(i) for i in range(10)]
        reference_result = loadmat('./v1like_ref/reference_v1like_result.mat')['feature_matrix']
        old_images = loadmat('./v1like_ref/reference_v1like_result.mat')['images_after_resize']
        # now let's get them.
        # use default parameters.
        # get the fitlers.
        pars = v1like.default_pars()
        filt_l, filt_l_raw = v1s_misc._get_gabor_filters(pars['representation']['filter'])
        result = []
        filt_l_raw_reconstruct = [np.sum(np.array([row.dot(col) for row, col in filt_l[i]]), axis=0) for i in
                                  range(len(filt_l))]
        with Timer('faster version'):
            for idx, imagename in enumerate(image_list):
                im_legacy = old_images[idx]

                result.append(v1like._part_generate_repr(im_legacy, pars['steps'] - {'preproc_resize'},
                                                         pars['representation'], pars['featsel'], filt_l_raw_reconstruct,
                                                         legacy=False, debug=True))
                print("finish image {}".format(idx + 1))

        result_legacy = np.array(result)
        self.assertEqual(reference_result.dtype, result_legacy.dtype)
        self.assertEqual(reference_result.shape, result_legacy.shape)
        # these two parts are pretty different.
        self.assertTrue(np.percentile(abs(reference_result[:, :30 * 30 * 96] - result_legacy[:, :30 * 30 * 96]),
                                      99) < 1e-3)
        self.assertTrue(np.percentile(abs(reference_result[:, 30 * 30 * 96:] - result_legacy[:, 30 * 30 * 96:]),
                                      99) <= 1)

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
        for _ in range(10):
            image = self.get_new_image()
            kshape_this = np.random.choice([3, 5, 7, 9, 11])
            with Timer('low pass old'):
                image_processed_1 = v1like._preproc_lowpass(image, 3, True)
            with Timer('low pass new'):
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
            with Timer('local norm 2d new'):
                image_processed_2 = v1like._normin(image, params, False).astype(np.float32)
            with Timer('local norm 2d old'):
                image_processed_1 = v1like._normin(image, params, True).astype(np.float32)
            # t2 = time()
            # print('new take {}'.format(t2 - t1))
            # print(shape_this)
            # print(image_processed_1.shape, image_processed_2.shape)
            self.assertTrue(np.allclose(image_processed_1, image_processed_2, atol=tol))
            self.assertEqual(image_processed_1.shape, image_processed_2.shape)

    def test_local_normalization_3d(self):
        for _ in range(10):
            image = self.get_new_image(ndim=3)
            kshape_this = np.random.choice([3, 5, 7, 9, 11])
            threshold_this = np.random.rand() * 10.0
            params = {'kshape': (kshape_this, kshape_this), 'threshold': threshold_this}
            # t1 = time()

            # t2 = time()
            # print('old take {}'.format(t2-t1))
            # t1 = time()
            with Timer('local norm 3d new'):
                image_processed_2 = v1like._normin(image, params, False).astype(np.float32)
            with Timer('local norm 3d old'):
                image_processed_1 = v1like._normin(image, params, True).astype(np.float32)
            # t2 = time()
            # print('new take {}'.format(t2 - t1))
            # print(shape_this)
            # print(image_processed_1.shape, image_processed_2.shape)
            self.assertTrue(np.allclose(image_processed_1, image_processed_2, atol=tol))
            self.assertEqual(image_processed_1.shape, image_processed_2.shape)
            # print(abs(image_processed_1 - image_processed_2).max())

    def test_rdim(self):
        for _ in range(10):
            for lsum_ksize in [1, 3, 5, 7, 9]:
                image = self.get_new_image(3)
                outshape = np.random.randint(10, 20, (2,), dtype=np.uint16)
                with Timer('local norm 2d old'):
                    image_processed_1 = v1like._dimr(image, lsum_ksize, outshape, True)
                with Timer('local norm 2d new'):
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
            if debug:
                print('filter size {} x {}'.format(kshape, kshape))
            filt_l, filt_l_raw = v1s_misc._get_gabor_filters(filter_params)
            for _ in range(5):
                image = self.get_new_image()
                if debug:
                    print('image size: {}'.format(image.shape))
                normalize_vector_inplace(image)
                with Timer('filter legacy'):
                    conv_result_1 = v1like._filter(image, filt_l, True)
                with Timer('filter new'):
                    conv_result_2 = v1like._filter(image, filt_l_raw, False)
                self.assertTrue(np.allclose(conv_result_1, conv_result_2, atol=tol))
                self.assertEqual(conv_result_1.shape, conv_result_2.shape)


if __name__ == '__main__':
    unittest.main()
