import unittest
from early_vision_toolbox.v1like import v1like
import numpy as np
#from time import time

class MyTestCase(unittest.TestCase):
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
            shape_this = np.random.randint(50, 200, (2,), dtype=np.uint16)
            image = np.random.randn(shape_this[0], shape_this[1])
            image_processed_1 = v1like._preproc_lowpass(image, 3, True)
            image_processed_2 = v1like._preproc_lowpass(image, 3, False)
            self.assertTrue(np.allclose(image_processed_1, image_processed_2))

    def test_local_normalization(self):
        for _ in range(10):
            shape_this = np.random.randint(50, 200, (2,), dtype=np.uint16)
            image = np.random.randn(shape_this[0], shape_this[1])
            kshape_this = np.random.choice([3, 5, 7, 9, 11])
            threshold_this = np.random.rand()*10.0
            params = {'kshape': (kshape_this, kshape_this), 'threshold': threshold_this}
            # t1 = time()
            image_processed_1 = v1like._normin(image, params, True)
            # t2 = time()
            # print('old take {}'.format(t2-t1))
            # t1 = time()
            image_processed_2 = v1like._normin(image, params, False)
            # t2 = time()
            # print('new take {}'.format(t2 - t1))
            # print(shape_this)
            # print(image_processed_1.shape, image_processed_2.shape)
            # print(abs(image_processed_1 - image_processed_2).max())
            self.assertTrue(np.allclose(image_processed_1, image_processed_2))



if __name__ == '__main__':
    unittest.main()
