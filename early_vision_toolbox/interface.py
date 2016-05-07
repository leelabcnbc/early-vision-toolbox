# coding=utf-8
"""interfaces defined for many functions in the toolbox"""
from __future__ import division, print_function, absolute_import
import abc


class NeuronBank(object):
    """
    abstract class for a bank of neurons.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def n_neuron(self):
        """ number of neurons

        Returns
        -------
        number of neurons.
        """
        pass

    @abc.abstractproperty
    def rf_size(self):
        """ size of RF

        Returns
        -------
        size of filter, in (height, width).

        """
        pass

    @abc.abstractmethod
    def predict(self, imgs, neuron_idx=None):
        """ get neuron response to images

        Parameters
        ----------
        imgs: iterable
            an interable, each one being an image which the neuron can give response.

        Returns
        -------
        a 2d ndarray each row being a neuron's response to each image.

        """
        pass