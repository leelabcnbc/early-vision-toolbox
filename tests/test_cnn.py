from __future__ import division, absolute_import, print_function
import unittest
from early_vision_toolbox.cnn import caffe_models
from early_vision_toolbox.cnn import dir_dict
from early_vision_toolbox.cnn.network_definitions import caffe_deploy_proto_predefined, get_prototxt, net_info_dict
from early_vision_toolbox.cnn.caffe_models import fill_weights, create_empty_net, create_predefined_net
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




if __name__ == '__main__':
    unittest.main()
