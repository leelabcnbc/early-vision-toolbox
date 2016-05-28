from __future__ import absolute_import, division, print_function
import unittest
from early_vision_toolbox.v1like import v1like
from early_vision_toolbox.v1like.legacy import v1s_misc, v1s_funcs
from early_vision_toolbox.util import normalize_vector_inplace
import numpy as np
from scipy.io import loadmat
from skimage.io import imread
import time
from sklearn.pipeline import FeatureUnion

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
    def test_reference_plus_legacy(self):
        """compare with reference result of original implementation"""
        image_list = ['./v1like_ref/sample_{}.png'.format(i) for i in range(10)]
        reference_result = loadmat('./v1like_ref/reference_v1like_result_plus.mat')['feature_matrix']
        X = [imread(imagename) for imagename in image_list]
        v1like_instance = v1like.V1Like(pars_baseline='simple_plus', legacy=True, debug=debug)
        with Timer('simple_plus legacy version'):
            result_legacy = v1like_instance.transform(X)
        self.assertEqual(reference_result.dtype, result_legacy.dtype)
        self.assertEqual(reference_result.shape, result_legacy.shape)
        if debug:
            print(abs(reference_result[:, :] - result_legacy[:, :]).max())
        self.assertTrue(np.allclose(reference_result, result_legacy, atol=tol))

    def test_reference_plusplus_legacy(self):
        """compare with reference result of original implementation"""
        image_list = ['./v1like_ref/sample_{}.png'.format(i) for i in range(10)]
        reference_result = loadmat('./v1like_ref/reference_v1like_result_plusplus.mat')['feature_matrix']
        X = [imread(imagename) for imagename in image_list]
        v1like_instance_1 = v1like.V1Like(pars_baseline='simple_plus', legacy=True, debug=debug)
        v1like_instance_2 = v1like.V1Like(pars_baseline='simple_plusplus_2nd_scale', legacy=True, debug=debug)
        v1like_instance = FeatureUnion([('scale_1', v1like_instance_1),
                                        ('scale_2', v1like_instance_2)])
        # seems that FeatureUnion's X can't be a iterator. must be a true array.
        with Timer('simple_plus legacy version'):
            result_legacy = v1like_instance.transform(X)
        self.assertEqual(reference_result.dtype, result_legacy.dtype)
        self.assertEqual(reference_result.shape, result_legacy.shape)
        if debug:
            print(abs(reference_result[:, :] - result_legacy[:, :]).max())
        self.assertTrue(np.allclose(reference_result, result_legacy, atol=tol))

    def test_reference_simple_legacy(self):
        """compare with reference result of original implementation"""
        image_list = ['./v1like_ref/sample_{}.png'.format(i) for i in range(10)]
        reference_result = loadmat('./v1like_ref/reference_v1like_result.mat')['feature_matrix']
        X = [imread(imagename) for imagename in image_list]
        v1like_instance = v1like.V1Like(pars_baseline='simple', legacy=True, debug=debug)
        with Timer('simple_plus legacy version'):
            result_legacy = v1like_instance.transform(X)
        self.assertEqual(reference_result.dtype, result_legacy.dtype)
        self.assertEqual(reference_result.shape, result_legacy.shape)
        if debug:
            print(abs(reference_result[:, :] - result_legacy[:, :]).max())
        self.assertTrue(np.allclose(reference_result, result_legacy, atol=tol))

    def test_reference_plus_legacy_faster(self):
        """compare with reference result of original implementation"""
        reference_result = loadmat('./v1like_ref/reference_v1like_result_plus.mat')['feature_matrix']
        old_images = loadmat('./v1like_ref/reference_v1like_result_plus.mat')['images_after_resize']
        v1like_instance = v1like.V1Like(pars_baseline='simple_plus', legacy=False, debug=debug)
        pars = v1like.default_pars(type='simple')
        filt_l, _ = v1s_misc._get_gabor_filters(pars['representation']['filter'])
        filt_l_raw_reconstruct = [np.sum(np.array([row.dot(col) for row, col in filt_l[i]]), axis=0) for i in
                                  range(len(filt_l))]
        v1like_instance.reload_filters(filt_l_raw_reconstruct)
        with Timer('simple_plus faster version'):
            result_legacy = v1like_instance.transform(old_images)

        self.assertEqual(reference_result.dtype, result_legacy.dtype)
        self.assertEqual(reference_result.shape, result_legacy.shape)
        # these two parts are pretty different.
        self.assertTrue(np.percentile(abs(reference_result[:, :30 * 30 * 96] - result_legacy[:, :30 * 30 * 96]),
                                      99) < 1e-3)
        self.assertTrue(np.percentile(abs(reference_result[:, 30 * 30 * 96:] - result_legacy[:, 30 * 30 * 96:]),
                                      99) <= 1)

    def test_reference_simple_legacy_faster(self):
        """compare with reference result of original implementation"""
        """compare with reference result of original implementation"""
        reference_result = loadmat('./v1like_ref/reference_v1like_result.mat')['feature_matrix']
        old_images = loadmat('./v1like_ref/reference_v1like_result.mat')['images_after_resize']
        v1like_instance = v1like.V1Like(pars_baseline='simple', legacy=False, debug=debug)
        pars = v1like.default_pars(type='simple')
        filt_l, _ = v1s_misc._get_gabor_filters(pars['representation']['filter'])
        filt_l_raw_reconstruct = [np.sum(np.array([row.dot(col) for row, col in filt_l[i]]), axis=0) for i in
                                  range(len(filt_l))]
        v1like_instance.reload_filters(filt_l_raw_reconstruct)
        with Timer('simple faster version'):
            result_legacy = v1like_instance.transform(old_images)

        self.assertEqual(reference_result.dtype, result_legacy.dtype)
        self.assertEqual(reference_result.shape, result_legacy.shape)
        # these two parts are pretty different.
        self.assertTrue(np.percentile(abs(reference_result - result_legacy), 99) < 1e-3)

    def test_reference_plusplus_legacy_faster(self):
        """compare with reference result of original implementation"""
        reference_result = loadmat('./v1like_ref/reference_v1like_result_plusplus.mat')['feature_matrix']
        old_images = loadmat('./v1like_ref/reference_v1like_result_plusplus.mat')['images_after_resize']
        old_images_2 = loadmat('./v1like_ref/reference_v1like_result_plusplus.mat')['images_after_resize_2']
        v1like_instance_1 = v1like.V1Like(pars_baseline='simple_plus', legacy=False, debug=debug)
        v1like_instance_2 = v1like.V1Like(pars_baseline='simple_plusplus_2nd_scale', legacy=False, debug=debug)
        pars = v1like.default_pars(type='simple')
        filt_l, _ = v1s_misc._get_gabor_filters(pars['representation']['filter'])
        filt_l_raw_reconstruct = [np.sum(np.array([row.dot(col) for row, col in filt_l[i]]), axis=0) for i in
                                  range(len(filt_l))]
        v1like_instance_1.reload_filters(filt_l_raw_reconstruct)
        v1like_instance_2.reload_filters(filt_l_raw_reconstruct)
        with Timer('simple_plus faster version'):
            result_legacy_1 = v1like_instance_1.transform(old_images)
            result_legacy_2 = v1like_instance_2.transform(old_images_2)
        result_legacy = np.concatenate((result_legacy_1, result_legacy_2), axis=1)

        self.assertEqual(reference_result.dtype, result_legacy.dtype)
        self.assertEqual(reference_result.shape, result_legacy.shape)
        # these two parts are pretty different.
        slice_scale_1_simple = slice(None, 30 * 30 * 96)
        slice_scale_1_complex = slice(30 * 30 * 96, 126400)
        slice_scale_2_simple = slice(126400, 126400 + 30 * 30 * 96)
        slice_scale_2_complex = slice(126400 + 30 * 30 * 96, None)
        if debug:
            print(np.percentile(abs(reference_result[:, slice_scale_1_simple] - result_legacy[:, slice_scale_1_simple]),
                                99))
            print(np.percentile(abs(reference_result[:, slice_scale_2_simple] - result_legacy[:, slice_scale_2_simple]),
                                99))
            print(
                np.percentile(abs(reference_result[:, slice_scale_1_complex] - result_legacy[:, slice_scale_1_complex]),
                              99))
            print(
                np.percentile(abs(reference_result[:, slice_scale_2_complex] - result_legacy[:, slice_scale_2_complex]),
                              99))
        self.assertTrue(
            np.percentile(abs(reference_result[:, slice_scale_1_simple] - result_legacy[:, slice_scale_1_simple]),
                          99) < 1e-3)

        self.assertTrue(
            np.percentile(abs(reference_result[:, slice_scale_2_simple] - result_legacy[:, slice_scale_2_simple]),
                          99) < 1e-3)
        self.assertTrue(
            np.percentile(abs(reference_result[:, slice_scale_1_complex] - result_legacy[:, slice_scale_1_complex]),
                          99) <= 1)
        self.assertTrue(
            np.percentile(abs(reference_result[:, slice_scale_2_complex] - result_legacy[:, slice_scale_2_complex]),
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
