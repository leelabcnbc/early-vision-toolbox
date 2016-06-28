from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt


def scatter_with_image_markers(locations, images, markersize=1.0, axis_handle=None):
    assert locations.ndim == 2 and locations.shape[1] == 2
    assert len(images) == locations.shape[0]

    if axis_handle is None:
        axis_handle = plt.gca()
    # don't clear, so you can do hold on type of thing.
    # axis_handle.clear()
    xlim = axis_handle.get_xlim()
    ylim = axis_handle.get_ylim()
    for im_idx, image in enumerate(images):
        data_point = locations[im_idx]
        axis_handle.imshow(image, extent=[data_point[0] - markersize / 2, data_point[0] + markersize / 2,
                                          data_point[1] - markersize / 2, data_point[1] + markersize / 2])
    # otherwise, it won't look correct.
    axis_handle.set_xlim(xlim)
    axis_handle.set_ylim(ylim)