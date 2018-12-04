from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorlayer.layers import *
import tensorlayer as tl
import tensorflow as tf


def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    # print('#' + str(shape))
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    # print('*' + str(shape))
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


class MaxoutDenseLayer(Layer):

    def __init__(
            self,
            layer,
            n_units=100,
            act=tf.identity,
            branch=1,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='maxout',
    ):
        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        print("MaxoutDenseLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name):
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, **W_init_args)
            if b_init is not None:
                try:
                    b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, **b_init_args)
                except Exception:  # If initializer is a constant, do not specify shape.
                    b = tf.get_variable(name='b', initializer=b_init, dtype=tf.float32, **b_init_args)
                h1 = tf.matmul(self.inputs, W) + b
                self.h1 = h1
                self.outputs = max_out(h1, n_units)
            else:
                h1 = tf.matmul(self.inputs, W)
                self.h1 = h1
                self.outputs = max_out(h1,  n_units)

        # Hint : list(), dict() is pass by value (shallow), without them, it is
        # pass by reference.
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init is not None:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])

