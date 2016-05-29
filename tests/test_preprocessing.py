from __future__ import division, print_function, absolute_import
import unittest
import h5py
import numpy as np
from skimage.io import imread
from early_vision_toolbox.preprocessing import whiten_olsh_lee_inner, whole_image_preprocessing_pipeline


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

    def test_whole_image_preprocessing_pipeline(self):
        """test against legacy code for the whole pipeline"""

        # remember to transpose last two dimensions before comparison...
        reference_result_file = 'preprocessing_ref/rsa_research_preprocessing_ref_alexnet_sparsedbn.hdf5'
        f = h5py.File(reference_result_file, 'r')
        image_dir = 'preprocessing_ref/rsa_research_data/images/'

        # use f_ since f will overwrite f for HDF5... sucks Python 2
        img_ec_list = [imread(image_dir + f_) for f_ in ['ec_001_s0.png', 'ec_002_s0.png']]
        img_ac_list = [imread(image_dir + f_) for f_ in ['ac_001_s0.png', 'ac_002_s0.png']]
        img_ex_list = [imread(image_dir + f_) for f_ in ['ex_001_s0.png', 'ex_002_s0.png']]
        image_list = (img_ec_list, img_ac_list, img_ex_list)

        def test_rsa_research_preprocessing_one_group(grp_handle, jittermaxpixellist):
            # go over group by group.
            for type_name, img_list, canvas_color in zip(['ec', 'ac', 'ex'], image_list,
                                                         [128.0 / 255, 127.0 / 255, 127.0 / 255]):
                type_group_this = grp_handle[type_name]
                for size_type, scaling_this, jittermaxpixel in zip(['11', '22', '33'], [1.0 / 3, 2.0 / 3, 3.0 / 3],
                                                                   jittermaxpixellist):
                    size_group_this = type_group_this[size_type]
                    for x in size_group_this:
                        ref_data_this = size_group_this[x][:].transpose(0, 2, 1)
                        ref_data_this_jitterseed = size_group_this[x].attrs['jitterrandseed'].ravel()[0]
                        # print(ref_data_this.shape, ref_data_this_jitterseed)
                        # time to construct preprocessing.
                        steps = {'normalize_format', 'rescale', 'putInCanvas'}
                        pars = {'rescale': {'imscale': scaling_this,
                                            'order': 1},  # interpolation order. 1 means bilinear.
                                'putInCanvas': {'canvascolor': canvas_color,  # gray color by default.
                                                'jitter': True,  # no jitter
                                                'jittermaxpixel': jittermaxpixel,  # trivial jitter.
                                                'jitterrandseed': ref_data_this_jitterseed,
                                                'canvassize': ref_data_this.shape[-2:]
                                                },
                                }
                        pipeline_this, _ = whole_image_preprocessing_pipeline(steps, pars)
                        data_this = np.array(pipeline_this.transform(img_list))
                        self.assertEqual(data_this.ndim, 4)
                        self.assertEqual(data_this.shape[-1], 3)
                        self.assertTrue(np.array_equal(data_this[:, :, :, 0], data_this[:, :, :, 1]))
                        self.assertTrue(np.array_equal(data_this[:, :, :, 0], data_this[:, :, :, 2]))
                        self.assertTrue(np.allclose(data_this[:, :, :, 0], ref_data_this))
                        # print((data_this[:, :, :, 0]-ref_data_this).mean())
                        # for some reason, it's not all equal bit by bit. maybe due to some small changes in skimage
                        # across different versions.
                        # print(np.array_equal(data_this[:, :, :, 0], ref_data_this))

        # reference result for AlexNet.
        test_rsa_research_preprocessing_one_group(f['alexnet'], [2, 4, 6])

        # reference result for SparseDBN.
        test_rsa_research_preprocessing_one_group(f['sparsedbn'], [1, 2, 3])
        f.close()


if __name__ == '__main__':
    unittest.main()
