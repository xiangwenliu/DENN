from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorlayer.layers import *
from tensorflow.python.framework import ops
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import random


def init_weight(shape):
  return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
  return tf.Variable(tf.random_normal(shape))



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
    The :class:`MyDenseLayer` class is a dendrite connected layer.
    parameters:
    branch the number of branch
    oneper the percent of 1 in mask layer
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
            is_scale=False,
            is_train=False
    ):
        self.branch = None
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.branch = branch
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])
        # batchsize = int(self.inputs.get_shape()[0])

        # print('-------------------------------------------------------------n_in=', n_in)
        # print('-------------------------------------------------------------batchsize', batchsize)
        self.n_units = n_units
        print("  [TL] DendriteLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name) as vs:

            # get the mask
            mm = get_mask(self.branch, n_in, n_units)
            mask = tf.get_variable(name='mask', shape=[branch * n_units, n_in], trainable=False,
                    initializer=tf.constant_initializer(value=mm))

            W = tf.get_variable(name='W', shape=[self.n_units*self.branch, n_in],
                                initializer=W_init, **W_init_args)

            mask_weight = tf.multiply(mask, W)
            out = tf.tensordot(self.inputs, mask_weight, axes=[[1], [1]])
            #batchsize, branch*output array
            h1 = out
            
            self.output_h1 = h1
            
            h1 = tf.nn.max_pool(tf.reshape(h1, shape=[-1, self.branch * self.n_units, 1, 1]),
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
        # self.all_params.extend([bias1])
        if b_init is not None:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])



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
