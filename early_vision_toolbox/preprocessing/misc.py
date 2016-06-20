from __future__ import division, print_function, absolute_import
import numpy as np
from skimage.transform import rescale


def make_sure_rgb_image(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    assert img.ndim == 3 and img.shape[2] == 3
    return img


def make_sure_some_size(image_raw, smallest_size, msg=None):
    assert type(image_raw) == np.ndarray  # don't use isinstance to be more bulletproof, as subclass can
    # behave strangely sometimes.
    height, width = image_raw.shape[:2]
    if height <= smallest_size or width <= smallest_size:
        if height <= width:
            rescale_ratio = smallest_size / height
        else:
            rescale_ratio = smallest_size / width
        image_rescaled = rescale(image_raw, rescale_ratio, order=1, mode='edge')
        if msg is not None:
            print(msg)
        print('fix size, from {} to {}'.format(image_raw.shape, image_rescaled.shape))
    else:
        image_rescaled = image_raw.copy()
    assert image_rescaled.dtype == np.float64
    return image_rescaled
