"""size utility module for CNN. basically a rewrite of my old code to compute any size-related thing to CNN"""

from __future__ import division, print_function, absolute_import
import numpy as np
from collections import OrderedDict
from copy import deepcopy


def check_exact_cover(output_size, stride, kernelsize, input_size, pad_this):
    assert (output_size[0] - 1) * stride + kernelsize == input_size[0] + 2 * pad_this, "input not exactly covered"
    assert (output_size[1] - 1) * stride + kernelsize == input_size[1] + 2 * pad_this, "input not exactly covered"


class CNNSizeHelper(object):
    def __init__(self, layer_info_dict, input_size=None):
        """

        Parameters
        ----------
        layer_names
        layer_strides
        layer_kernelsizes
        layer_pads
        input_size
        """
        assert isinstance(layer_info_dict, OrderedDict)

        num_layer = len(layer_info_dict)
        for layer, value in layer_info_dict.iteritems():
            assert {'pad', 'stride', 'kernelsize'} == set(value.keys())

        # create "global" versions of layer_xxx variables, by computing layer by layer.
        layer_strides_g = [None] * num_layer
        layer_kernelsizes_g = [None] * num_layer
        layer_pads_g = [None] * num_layer

        result_dict = OrderedDict()
        # input.
        last_stride_g = 1
        last_kernelsize_g = 1
        last_pad_g = 0  # cummulative padding.
        for layer, info_dict in layer_info_dict.iteritems():
            # for each layer, I have to check that the units in the output blob
            # can perfectly cover the units in the current blob (no uncovered units, nor redundant receptive field).
            # This is crucial for unpooling, which only works for such case for now.
            # In the long term, I think Caffe needs to implement deconv net itself.

            # compute the stride, kernelsize, and pad for this layer, w.r.t. input layer.
            stride_this = info_dict['stride']
            kernelsize_this = info_dict['kernelsize']
            pad_this = info_dict['pad']

            pad_this_g = pad_this * last_stride_g + last_pad_g  # total padding in input layer
            stride_this_g = stride_this * last_stride_g  # total stride in input layer.
            kernelsize_this_g = (kernelsize_this - 1) * last_stride_g + last_kernelsize_g

            info_dict_out_this = deepcopy(info_dict)
            info_dict_out_this.update({
                'stride_g': stride_this_g,
                'kernelsize_g': kernelsize_this_g,
                'pad_g': pad_this_g
            })
            result_dict[layer] = info_dict_out_this

            last_stride_g = stride_this_g
            last_kernelsize_g = kernelsize_this_g
            last_pad_g = pad_this_g
        self.layer_info_dict = result_dict

        if input_size is not None:
            self.compute_output_size(input_size)

    @property
    def output_size_dict(self):
        """for some legacy tests"""
        return {layer: self.layer_info_dict[layer]['output_size'] for layer in self.layer_info_dict}

    @property
    def field_size_dict(self):
        """for some legacy tests"""
        return {layer: self.layer_info_dict[layer]['kernelsize_g'] for layer in self.layer_info_dict}

    @property
    def stride_dict(self):
        """for some legacy tests"""
        return {layer: self.layer_info_dict[layer]['stride_g'] for layer in self.layer_info_dict}

    @property
    def pad_dict(self):
        """for some legacy tests"""
        return {layer: self.layer_info_dict[layer]['pad_g'] for layer in self.layer_info_dict}

    def compute_output_size(self, input_size):
        assert len(input_size) == 2, "you must specify both height and width"
        last_input_size = input_size
        for layer, info_dict_this in self.layer_info_dict.iteritems():
            pad_this = info_dict_this['pad']
            stride_this = info_dict_this['stride']
            kernelsize_this = info_dict_this['kernelsize']

            pad_this_g = info_dict_this['pad_g']
            stride_this_g = info_dict_this['stride_g']
            kernelsize_this_g = info_dict_this['kernelsize_g']

            # compute the dimension of units for output.
            output_size = ((last_input_size[0] + 2 * pad_this - kernelsize_this) // stride_this + 1,
                           (last_input_size[1] + 2 * pad_this - kernelsize_this) // stride_this + 1)

            # check that this output check can indeed exactly cover the input layer below,
            check_exact_cover(output_size, stride_this, kernelsize_this, last_input_size, pad_this)
            # as well as input image (plus padding).
            check_exact_cover(output_size, stride_this_g, kernelsize_this_g, input_size, pad_this_g)
            input_size_this = last_input_size
            input_size_this_g = (input_size[0] + 2 * pad_this_g, input_size[1] + 2 * pad_this_g)

            info_dict_this.update({
                'output_size': output_size,
                'input_size': input_size_this,
                'input_size_g': input_size_this_g
            })

            last_input_size = output_size
        self.input_size = input_size

    def compute_range(self, layer_name, top_left, bottom_right):
        """ everything here is exclusive.

        Parameters
        ----------
        layer_name
        top_left
        bottom_right

        Returns
        -------

        """
        top_left = np.asarray(top_left, dtype=int)
        bottom_right = np.asarray(bottom_right, dtype=int) - 1  # this is not in place.
        assert top_left.shape == bottom_right.shape == (2,)
        assert np.all(top_left <= bottom_right) and np.all(top_left >= 0)

        layer_info_dict_this = self.layer_info_dict[layer_name]
        field_size = layer_info_dict_this['kernelsize_g']
        stride = layer_info_dict_this['stride_g']
        pad = layer_info_dict_this['pad_g']

        top_left_output = stride * top_left[0] - pad, stride * top_left[1] - pad
        bottom_right_output = stride * bottom_right[0] + field_size - pad, stride * bottom_right[1] + field_size - pad

        return top_left_output, bottom_right_output

    def compute_valid_area(self, layer_name, row, col):
        """ gives the indices in the input image space of one unit's receptive field.
        :param layer_name:
        :type layer_name: string
        :param row:
        :type row: int
        :param col:
        :type col: int
        :return: (range_raw_r, range_raw_c, range_valid_r, range_valid_c). range_raw_r/c are raw indices (may contain
        negative or too big values), and range_valid_r/c are the boolean mask of valid indices (True being valid).
        :rtype:
        """
        layer_info_dict_this = self.layer_info_dict[layer_name]

        field_size = layer_info_dict_this['kernelsize_g']
        stride = layer_info_dict_this['stride_g']
        pad = layer_info_dict_this['pad_g']
        range_raw_r = np.arange(stride * row - pad, stride * row + field_size - pad).astype(np.int)
        range_raw_c = np.arange(stride * col - pad, stride * col + field_size - pad).astype(np.int)
        range_valid_r = np.logical_and(range_raw_r >= 0, range_raw_r < self.input_size[0])
        range_valid_c = np.logical_and(range_raw_c >= 0, range_raw_c < self.input_size[1])

        return range_raw_r, range_raw_c, range_valid_r, range_valid_c

    def compute_inside_neuron(self, layer_name):
        """ compute the range of column and row indices that give a neuron whose RF is completely in the image.

        the ranges are given in Python convention, exclusive on the right part.
        Parameters
        ----------
        layer_name

        Returns
        -------

        """
        layer_info_dict_this = self.layer_info_dict[layer_name]

        field_size = layer_info_dict_this['kernelsize_g']
        stride = layer_info_dict_this['stride_g']
        pad = layer_info_dict_this['pad_g']

        input_rows = self.input_size[0]
        input_cols = self.input_size[1]
        row_range = (np.ceil(1.0 * pad / stride), np.floor(1.0 * (input_rows - field_size + pad) / stride))
        col_range = (np.ceil(1.0 * pad / stride), np.floor(1.0 * (input_cols - field_size + pad) / stride))
        if not (row_range[0] <= row_range[1] and col_range[0] <= col_range[1]):
            raise ValueError('No inside neuron!')
        return row_range[0], row_range[1] + 1, col_range[0], col_range[1] + 1

    def compute_minimum_coverage(self, layer_name, top_left, bottom_right, one_over_in=True, one_over_out=True):
        """ compute the miminum grid of neurons that can cover a rectangle with top left at top_left,
        and bottom right at bottom_right

        here everything is exclusive by default.

        Parameters
        ----------
        layer_name
        top_left
        bottom_right

        Returns
        -------

        """
        # use int to avoid any potential problem. look that I floor the top left and ceil the bottom right to make sure
        # the actual image is covered, even with float input.
        top_pos, left_pos = np.floor(np.asarray(top_left)).astype(np.int)
        bottom_pos, right_pos = np.ceil(np.asarray(bottom_right)).astype(np.int) - (1 if one_over_in else 0)
        assert 0 <= top_pos <= bottom_pos < self.input_size[0]
        assert 0 <= left_pos <= right_pos < self.input_size[1]

        # brutal force, find the four extreme neurons. It's fine, as we don't need performance here.
        output_size_this_layer = self.layer_info_dict[layer_name]['output_size']
        # get the coverage of all neurons.
        row_idx, col_idx = np.meshgrid(np.arange(output_size_this_layer[0]), np.arange(output_size_this_layer[1]),
                                       indexing='ij')
        stride_g = self.layer_info_dict[layer_name]['stride_g']
        pad_g = self.layer_info_dict[layer_name]['pad_g']
        kernelsize_g = self.layer_info_dict[layer_name]['kernelsize_g']

        min_r_all = stride_g * row_idx - pad_g
        max_r_all = stride_g * row_idx - pad_g + kernelsize_g

        min_c_all = stride_g * col_idx - pad_g
        max_c_all = stride_g * col_idx - pad_g + kernelsize_g

        mask_topleft = np.logical_and(np.logical_and(min_r_all <= top_pos, top_pos < max_r_all),
                                      np.logical_and(min_c_all <= left_pos, left_pos < max_c_all))
        top_left_loc = row_idx[mask_topleft].max(), col_idx[mask_topleft].max()
        assert mask_topleft[top_left_loc]

        mask_bottomright = np.logical_and(np.logical_and(min_r_all <= bottom_pos, bottom_pos < max_r_all),
                                          np.logical_and(min_c_all <= right_pos, right_pos < max_c_all))
        bottom_right_loc = row_idx[mask_bottomright].min(), col_idx[mask_bottomright].min()
        assert mask_bottomright[bottom_right_loc]
        #bottom_right_loc_plus_one = bottom_right_loc[0] + 1, bottom_right_loc[1] + 1
        # either we have ordinary case
        # or we simply pick the central one.

        # ordinary_flag = top_left_loc[0] <= bottom_right_loc[0] and top_left_loc[1] <= bottom_right_loc[1]
        # toobig_flag = top_left_loc[0] > bottom_right_loc[0] and top_left_loc[1] > bottom_right_loc[1]
        # if not (ordinary_flag or toobig_flag):
        #     print(top_left_loc, bottom_right_loc, top_left, bottom_right)
        #     print('top left:', min_r_all[top_left_loc], min_c_all[top_left_loc], max_r_all[top_left_loc],
        #           max_c_all[top_left_loc])
        #     print('bottom_right left:', min_r_all[bottom_right_loc], min_c_all[bottom_right_loc], max_r_all[bottom_right_loc],
        #           max_c_all[bottom_right_loc])
        #     print('=========')
        # assert ordinary_flag or toobig_flag
        # if ordinary_flag:
        #     return top_left_loc, (bottom_right_loc_plus_one if one_over_out else bottom_right_loc)
        # else:
        #     assert toobig_flag
        #     central_loc = (top_left_loc[0] + bottom_right_loc[0]) // 2, (top_left_loc[1] + bottom_right_loc[1]) // 2
        #     assert mask_topleft[central_loc] and mask_bottomright[central_loc]
        #     central_loc_plus_one = central_loc[0] + 1, central_loc[1] + 1
        #     return central_loc, (central_loc_plus_one if one_over_out else central_loc)

        # it's not the case that it's either ordinary or too big.
        # first, check top_left_loc[0] and bottom_right_loc[0]
        if top_left_loc[0] <= bottom_right_loc[0]:
            # ordinary case
            tl_loc_0 = top_left_loc[0]
            br_loc_0 = bottom_right_loc[0] + (1 if one_over_out else 0)
        else:
            # the choice of neuron is ambiguous. simply pick the central one (well this central one is floored if
            # there's no exact central to be chosen).
            tl_loc_0 = (top_left_loc[0] + bottom_right_loc[0])//2
            br_loc_0 = tl_loc_0 + (1 if one_over_out else 0)

        if top_left_loc[1] <= bottom_right_loc[1]:
            tl_loc_1 = top_left_loc[1]
            br_loc_1 = bottom_right_loc[1] + (1 if one_over_out else 0)
        else:
            tl_loc_1 = (top_left_loc[1] + bottom_right_loc[1]) // 2
            br_loc_1 = tl_loc_1 + (1 if one_over_out else 0)

        return (tl_loc_0, tl_loc_1), (br_loc_0, br_loc_1)


layer_info_dict_raw = {
    # in the order of layer name, stride, kernelsize, pad.
    'vgg16': [('conv1_1', 1, 3, 1),
              ('conv1_2', 1, 3, 1),
              ('pool1', 2, 2, 0),
              ('conv2_1', 1, 3, 1),
              ('conv2_2', 1, 3, 1),
              ('pool2', 2, 2, 0),
              ('conv3_1', 1, 3, 1),
              ('conv3_2', 1, 3, 1),
              ('conv3_3', 1, 3, 1),
              ('pool3', 2, 2, 0),
              ('conv4_1', 1, 3, 1),
              ('conv4_2', 1, 3, 1),
              ('conv4_3', 1, 3, 1),
              ('pool4', 2, 2, 0),
              ('conv5_1', 1, 3, 1),
              ('conv5_2', 1, 3, 1),
              ('conv5_3', 1, 3, 1),
              ('pool5', 2, 2, 0)],
    'vgg19': [('conv1_1', 1, 3, 1),
              ('conv1_2', 1, 3, 1),
              ('pool1', 2, 2, 0),
              ('conv2_1', 1, 3, 1),
              ('conv2_2', 1, 3, 1),
              ('pool2', 2, 2, 0),
              ('conv3_1', 1, 3, 1),
              ('conv3_2', 1, 3, 1),
              ('conv3_3', 1, 3, 1),
              ('conv3_4', 1, 3, 1),
              ('pool3', 2, 2, 0),
              ('conv4_1', 1, 3, 1),
              ('conv4_2', 1, 3, 1),
              ('conv4_3', 1, 3, 1),
              ('conv4_4', 1, 3, 1),
              ('pool4', 2, 2, 0),
              ('conv5_1', 1, 3, 1),
              ('conv5_2', 1, 3, 1),
              ('conv5_3', 1, 3, 1),
              ('conv5_4', 1, 3, 1),
              ('pool5', 2, 2, 0)],
    'alexnet': [('conv1', 4, 11, 0),
                ('norm1', 1, 1, 0),
                ('pool1', 2, 3, 0),
                ('conv2', 1, 5, 2),
                ('norm2', 1, 1, 0),
                ('pool2', 2, 3, 0),
                ('conv3', 1, 3, 1),
                ('conv4', 1, 3, 1),
                ('conv5', 1, 3, 1),
                ('pool5', 2, 3, 0)],
    'caffenet': [('conv1', 4, 11, 0),
                 ('pool1', 2, 3, 0),
                 ('norm1', 1, 1, 0),
                 ('conv2', 1, 5, 2),
                 ('pool2', 2, 3, 0),
                 ('norm2', 1, 1, 0),
                 ('conv3', 1, 3, 1),
                 ('conv4', 1, 3, 1),
                 ('conv5', 1, 3, 1),
                 ('pool5', 2, 3, 0)]
}


def create_info_dict(info_dict_raw):
    info_dict = OrderedDict()
    for layer, stride, kernelsize, pad in info_dict_raw:
        info_dict[layer] = {
            'stride': stride,
            'kernelsize': kernelsize,
            'pad': pad
        }
    return info_dict


layer_info_dict_dict = {key: create_info_dict(value) for key, value in layer_info_dict_raw.iteritems()}


def create_size_helper(name_or_dict, input_size=None, last_layer=None):
    if isinstance(name_or_dict, str):
        info_dict = deepcopy(layer_info_dict_dict[name_or_dict])
    else:
        info_dict = deepcopy(name_or_dict)
    if last_layer is not None:
        keys_all = list(info_dict.keys())
        first_key_to_remove_idx = keys_all.index(last_layer)
        for layer_to_remove in keys_all[first_key_to_remove_idx+1:]:
            del info_dict[layer_to_remove]
    return CNNSizeHelper(info_dict, input_size)
