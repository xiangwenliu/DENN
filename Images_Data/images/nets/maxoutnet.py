from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorlayer.layers import *
import tensorlayer as tl
import tensorflow as tf

# build maxout neural network
def maxoutnet(images, num_classes=10, is_training=False,
            factor=0.1,
            reuse=True,
            nlayer=3,
            branch=2,
            netlen=512,
            keep_prob=0.8,
            l2_scales=0.0,
            normalization='No',
            scope='MaxoutFullConnectNet'):
    W_init2 = tf.uniform_unit_scaling_initializer(factor=factor)
    end_points = {}

    with tf.variable_scope(scope, 'MaxoutFullConnectNet', [images, num_classes]):
        tl.layers.set_name_reuse(reuse)
        net = InputLayer(images, name='input')
        net = FlattenLayer(net, name='flatten')
        n_input = net.outputs.get_shape()[-1]
        for layer in range(nlayer - 1):
            if layer % 2 == 1:
                net = MaxoutDenseLayer(net, n_units=netlen*branch, act=max_out, branch=branch,
                                       W_init=W_init2, name='fc' + str(layer + 1))
            else:
                net = MaxoutDenseLayer(net, n_units=netlen, act=max_out, branch=branch,
                                       W_init=W_init2, name='fc' + str(layer + 1))
            if layer > 0:
                net = tl.layers.DropoutLayer(net, keep=keep_prob,is_fix=True,
                                                 is_train=is_training, name='drop'+str(layer+1))
            if normalization == 'LN':
                net = tl.layers.LayerNormLayer(net, act=tf.nn.relu, trainable=is_training, name='LN' + str(layer + 1))
            if normalization == 'BN':
                net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_training, name='BN' + str(layer + 1))


        logits = DenseLayer(net, n_units=num_classes, act=tf.identity, W_init=W_init2, name='fc' + str(nlayer))

        L2 = 0.0

    end_points['layer'] = logits.all_layers;
    end_points['param'] = logits.all_params;

    return logits.outputs, end_points, L2


def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
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
                self.outputs = max_out(tf.matmul(self.inputs, W) + b, n_units//branch)
            else:
                self.outputs = max_out(tf.matmul(self.inputs, W),  n_units//branch)

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

