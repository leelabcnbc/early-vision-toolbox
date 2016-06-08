from __future__ import division, absolute_import, print_function
import unittest
from early_vision_toolbox.cnn import caffe_models
from early_vision_toolbox.cnn import dir_dict
from early_vision_toolbox.cnn.network_definitions import caffe_deploy_proto_predefined, get_prototxt, net_info_dict
from early_vision_toolbox.cnn.caffe_models import fill_weights, create_empty_net, create_predefined_net
from early_vision_toolbox.cnn.size_util import create_size_helper
from unittest import skip
import numpy as np


# TODO if we really want to be super sure, we should compile CPU caffe on travis and setup CI for this project.

class MyTestCase(unittest.TestCase):
    @skip('useless')
    def test_caffe_init(self):
        net = caffe_models.create_empty_net('alexnet', 'pool5')
        for layer_name, param in net.params.iteritems():
            print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))
        for x in dir_dict:
            print(x, dir_dict[x])

    def test_caffe_protosplit(self):
        for key, value in caffe_deploy_proto_predefined.items():
            result, net_base = value
            prototxt_reconstruct = ''.join([net_base] + result.values())
            prototxt_original = get_prototxt(net_info_dict[key][0])
            self.assertEqual(prototxt_reconstruct, prototxt_original)

    def test_caffe_fill(self):
        for key in caffe_deploy_proto_predefined:
            net_ref = create_predefined_net(key)
            net_new = fill_weights(key, create_empty_net(key))
            # check they are the same.
            assert set(net_ref.params.keys()) == set(net_new.params.keys())
            for layer_name, param in net_new.params.iteritems():
                assert len(param) == len(net_ref.params[layer_name])
                for idx in range(len(param)):
                    self.assertTrue(np.array_equal(param[idx].data, net_ref.params[layer_name][idx].data))

    def test_size_util_legacy(self):
        """ original size related tests.

        Returns
        -------

        """
        # first, a legacy alexnet test
        test_alexnet_projection()

        # then, test the valid neuron stuff.
        input_size_dict = {'alexnet': (227, 227),
                           'vgg16': (224, 224),
                           'vgg19': (224, 224),
                           'caffenet': (227, 227)}

        for model in input_size_dict:
            print('test legacy model {}'.format(model))
            # raw_input('wait to continue')
            input_size = input_size_dict[model]
            projection_this = create_size_helper(model, input_size=input_size)
            test_valid_neurons(projection_this, list(projection_this.layer_info_dict.keys()))

    def test_size_util_minimum_coverage(self):

        input_size_dict = {'alexnet': (227, 227),
                           'vgg16': (224, 224),
                           'vgg19': (224, 224),
                           'caffenet': (227, 227)}

        for model in input_size_dict:
            input_size = input_size_dict[model]
            projection_this = create_size_helper(model, input_size=input_size)
            print('testing model {}'.format(model))
            for layer in projection_this.layer_info_dict:
                print('testing model {}, layer {}'.format(model, layer))
                for idx in range(10):
                    print('iter {}'.format(idx + 1))
                    center = np.random.randint(100, input_size[0] - 100, size=(2,))
                    image_size = np.random.randint(50, 100)
                    # only square stuff tested yet.
                    top_left = center - image_size / 2
                    bottom_right = center + image_size / 2

                    tl_loc, br_loc = projection_this.compute_minimum_coverage(layer, top_left, bottom_right)

                    tl_loc = np.asarray(tl_loc, dtype=int)
                    br_loc = np.asarray(br_loc, dtype=int)

                    self.assertTrue(
                        test_size_util_minimum_coverage_helper(projection_this, layer, top_left, bottom_right,
                                                               tl_loc, br_loc))
                    # try removing one row
                    if br_loc[0] - tl_loc[0] > 1:
                        self.assertFalse(
                            test_size_util_minimum_coverage_helper(projection_this, layer, top_left, bottom_right,
                                                                   tl_loc + np.array([1, 0]), br_loc))
                        self.assertFalse(
                            test_size_util_minimum_coverage_helper(projection_this, layer, top_left, bottom_right,
                                                                   tl_loc, br_loc - np.array([1, 0])))
                    if br_loc[1] - tl_loc[1] > 1:
                        self.assertFalse(
                            test_size_util_minimum_coverage_helper(projection_this, layer, top_left, bottom_right,
                                                                   tl_loc + np.array([0, 1]), br_loc))
                        self.assertFalse(
                            test_size_util_minimum_coverage_helper(projection_this, layer, top_left, bottom_right,
                                                                   tl_loc, br_loc - np.array([0, 1])))


def test_size_util_minimum_coverage_helper(projection_this, layer, top_left, bottom_right, tl_loc, br_loc):
    # I have to verify that this is correct.
    # first, check that the region covered from top_left_loc to bottom_right_loc contains my ROI.
    range_original_tl, range_original_br = projection_this.compute_range(layer, tl_loc, br_loc)
    # print('area1: {}, {}; area 2: {}, {}'.format(top_left, bottom_right, range_original_tl, range_original_br))
    result = check_one_area_inside_another(top_left, bottom_right, range_original_tl, range_original_br)
    return result


def check_one_area_inside_another(top_left1, bottom_right1, top_left2, bottom_right2):
    """ check area 1 inside area 2. everything exclusive.

    Parameters
    ----------
    top_left1
    bottom_right1
    top_left2
    bottom_right2

    Returns
    -------

    """
    tb_flag = top_left2[0] <= top_left1[0] < bottom_right1[0] <= bottom_right2[0]
    lr_flag = top_left2[1] <= top_left1[1] < bottom_right1[1] <= bottom_right2[1]

    return tb_flag and lr_flag


def test_valid_neurons(helper, layer_names_alex):
    for layername in layer_names_alex:
        try:
            row_lower, row_upper, col_lower, col_upper = helper.compute_inside_neuron(layername)
        except ValueError as e:
            if e.args[0] == 'No inside neuron!':
                print('this layer {} has no inside neuron'.format(layername))
                # raw_input('wait to continue')
                continue
            else:
                raise e
        # check that these bounds are consistent with the ones computed by compute_valid_area
        print(layername, row_lower, row_upper, col_lower, col_upper)
        assert row_upper >= 1 and col_upper >= 1
        assert row_lower >= 0 and col_lower >= 0
        output_size_ref = helper.layer_info_dict[layername]['output_size']
        assert row_upper <= output_size_ref[0] and col_upper <= output_size_ref[1]
        # first check that neurons at (row_lower,col_lower) and (row_upper-1,col_upper-1) are with in RF.
        _, _, range_valid_r, range_valid_c = helper.compute_valid_area(layername, row_lower, col_lower)
        assert np.all(range_valid_r) and np.all(range_valid_c)

        _, _, range_valid_r, range_valid_c = helper.compute_valid_area(layername, row_upper - 1, col_upper - 1)
        assert np.all(range_valid_r) and np.all(range_valid_c)

        # then check that as long as we move out of the boundaries a little bit, we get invalid pixels.
        if row_lower > 0:
            _, _, range_valid_r, range_valid_c = helper.compute_valid_area(layername, row_lower - 1, col_lower)
            assert np.any(np.logical_not(range_valid_r)) and np.all(range_valid_c)

        if col_lower > 0:
            _, _, range_valid_r, range_valid_c = helper.compute_valid_area(layername, row_lower, col_lower - 1)
            assert np.any(np.logical_not(range_valid_c)) and np.all(range_valid_r)

        if row_upper < helper.output_size_dict[layername][0]:
            _, _, range_valid_r, range_valid_c = helper.compute_valid_area(layername, row_upper, col_upper - 1)
            assert np.any(np.logical_not(range_valid_r)) and np.all(range_valid_c)

        if col_upper < helper.output_size_dict[layername][1]:
            _, _, range_valid_r, range_valid_c = helper.compute_valid_area(layername, row_upper - 1, col_upper)
            assert np.any(np.logical_not(range_valid_c)) and np.all(range_valid_r)


def test_alexnet_projection():
    print("this is a program to check that the dictionaries for info of layers can be computed from old info")
    alex_net_projection = create_size_helper('alexnet', input_size=(227, 227))
    field_size_dict = {'conv1': 11, 'pool1': 19, 'conv2': 51, 'pool2': 67, 'conv3': 99, 'conv4': 131, 'conv5': 163,
                       'pool5': 195}
    stride_dict = {'conv1': 4, 'pool1': 8, 'conv2': 8, 'pool2': 16, 'conv3': 16, 'conv4': 16, 'conv5': 16, 'pool5': 32}
    pad_dict = {'conv1': 0, 'pool1': 0, 'conv2': 16, 'pool2': 16, 'conv3': 32, 'conv4': 48, 'conv5': 64, 'pool5': 64}
    key_to_select = list(field_size_dict.keys())
    # check they are the same.
    assert {key: alex_net_projection.field_size_dict[key] for key in key_to_select} == field_size_dict
    assert {key: alex_net_projection.stride_dict[key] for key in key_to_select} == stride_dict
    assert {key: alex_net_projection.pad_dict[key] for key in key_to_select} == pad_dict

    referenceDict = {'conv3': (13, 13),
                     'conv2': (27, 27),
                     'conv1': (55, 55),
                     'conv5': (13, 13),
                     'conv4': (13, 13),
                     'pool2': (13, 13),
                     'pool1': (27, 27),
                     'pool5': (6, 6)}

    assert {key: alex_net_projection.output_size_dict[key] for key in key_to_select} == referenceDict
    print("old results reproduced by computation!")


if __name__ == '__main__':
    unittest.main()
