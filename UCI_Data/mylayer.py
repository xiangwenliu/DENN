from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorlayer.layers import *
from tensorflow.python.framework import ops
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import random
import numbers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.eager import context


def init_weight(shape):
  return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
  return tf.Variable(tf.random_normal(shape))


def get_arry0(branch, inputlen, outputlen, percentage):
  return (np.random.rand(branch*outputlen, inputlen) < percentage).astype(np.float32)


def get_pre(branch, inputlen):
    arr = np.zeros((branch, inputlen))
    count = inputlen // branch
    remainder = inputlen % branch
    if count > 0:
        index = list(np.arange(inputlen))
        for i in range(branch):
            slice = random.sample(index, count)
            for item in slice:
                arr[i][item] = 1.
                index.remove(item)
        dx = random.sample(range(branch), remainder)
        for (x, item) in zip(dx, index):
            arr[x][item] = 1.
    else:
        dx = random.sample(range(branch), inputlen)
        dy = random.sample(range(inputlen), inputlen)
        for (x, y) in zip(dx, dy):
            arr[x][y] = 1.
    return arr

def get_random_mask(branch, inputlen,outputlen):
    pre = get_pre(branch,inputlen)
    for i in range(outputlen - 1):
        tem = get_pre(branch,inputlen)
        pre = np.concatenate((pre, tem), axis=0)
    return pre.astype(np.float32)

def get_arry1(branch,inputlen,outputlen,replace=False):
    mask = np.zeros([branch*outputlen,inputlen])
    section_size = int(inputlen/branch)
    #print(section_size)
    length = branch*section_size
    #print(length)
    def mapping(i):
        idx_map = np.random.choice(inputlen,length,replace=replace)
        idx_map = idx_map.reshape([branch,-1])
        #print(idx_map.shape)
        mask[(i*branch+np.arange(0,branch))[:,np.newaxis],idx_map]=1
    list(map(mapping,range(outputlen)))
    #print('range(outputlen)=',range(outputlen))
    return mask.astype(np.float32)

def get_mask(branch,inputlen,outputlen):
    if inputlen % branch == 0:
        return get_arry1(branch,inputlen,outputlen)
    else:
        return get_random_mask(branch,inputlen,outputlen)


class MyDenseLayer(Layer):
    """
    The :class:`MyDenseLayer` class is a dendritic neural network layer.
    parameters:
    branch: the number of branches per neuron
    keep_prob: dropout rate is 1-keep_prob
    n_units: the number of output units
    W_init: weight initialization
    """
    def __init__(
            self,
            layer=None,
            n_units=100,
            act=tf.identity,
            W_init=tf.uniform_unit_scaling_initializer(factor=0.1),#tf.truncated_normal_initializer(stddev=0.05),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args={},
            b_init_args={},
            name='my_dense_layer',
            branch=2,
            keep_prob=0.8,
            is_scale=False,
            is_train=False,
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.branch = branch
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape it")

        n_in = int(self.inputs.get_shape()[-1])

        self.n_units = n_units
        print("  [TL] DendriteLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name) as vs:
            mm = get_mask(self.branch, n_in, n_units)
            mask = tf.get_variable(name='mask', shape=[branch * n_units, n_in], trainable=False,
                    initializer=tf.constant_initializer(value=mm))
            W = tf.get_variable(name='W', shape=[self.n_units*self.branch, n_in],
                                initializer=W_init, **W_init_args)

            mask_weight = tf.multiply(mask, W)
            out = tf.tensordot(self.inputs, mask_weight, axes=[[1], [1]])
            # dropout
            if is_train is True and keep_prob < 1.0:
                out = dropout_unit(out,keep_prob=keep_prob)

            h1 = tf.nn.max_pool(tf.reshape(out, shape=[-1, self.branch * self.n_units, 1, 1]),
                                ksize=[1, self.branch, 1, 1], strides=[1, self.branch, 1, 1],
                                padding='VALID', data_format="NHWC")

            outputs = tf.reshape(h1, shape=[-1, self.n_units])

            if b_init is not None:
                try:
                    b = tf.get_variable(name='b', shape=(self.n_units), initializer=b_init, **b_init_args)
                except:  # If initializer is a constant, do not specify shape.
                    b = tf.get_variable(name='b', initializer=b_init, **b_init_args)
                self.outputs = act(outputs + b)
            else:
                self.outputs = act(outputs)

            if is_scale:
                self.outputs = self.outputs * self.branch

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

def dropout_unit(x,
            keep_prob=0.8,
            seed=None,
            noise_shape=None,
            name='dropout_unit',
            ):
    with ops.name_scope(name, "dropoutunit", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(
            keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        # Do nothing if we know keep_prob == 1
        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(
            noise_shape, seed=seed, dtype=x.dtype)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor
        if context.in_graph_mode():
            ret.set_shape(x.get_shape())
        return ret

class ScaleLayer(Layer):
    def __init__(
        self,
        layer=None,
        scale=1,
        b_init=tf.constant_initializer(value=0.0),
        name='scale_layer',
    ):
        # check layer name (fixed)
        Layer.__init__(self, name=name)

        # the input of this layer is the output of previous layer (fixed)
        self.inputs = layer.outputs

        self.outputs = self.inputs * scale

        # get stuff from previous layer (fixed)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        # update layer (customized)
        self.all_layers.extend( [self.outputs] )
