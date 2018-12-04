from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets import mylayer as mylayer
from tensorlayer.layers import *
import tensorlayer as tl
import tensorflow as tf

# build dendritic neural network
def dendritenet(images, num_classes=10, is_training=False,
               factor=0.1,
               reuse=True,
               nlayer=3,
               branch=2,
               netlen=512,
               keep_prob=0.8,
               l2_scales=0.0,
               normalization='No',
               scope='DendritNet'):
    W_init2 = tf.uniform_unit_scaling_initializer(factor=factor)
    end_points = {}
    act_func = tf.identity
    if normalization == 'No':
        act_func = tf.nn.relu
    with tf.variable_scope(scope, 'DendriteNet', [images, num_classes]):
        tl.layers.set_name_reuse(reuse)
        net = InputLayer(images, name='input')
        net = FlattenLayer(net, name='flatten')
        n_input = net.outputs.get_shape()[-1]
        for layer in range(nlayer-1):
            net = mylayer.MyDenseLayer(net, netlen, act=tf.identity, W_init=W_init2, branch=branch,keep_prob=keep_prob,
                                       is_scale=False, is_train=is_training, name='dfc'+str(layer+1))
            if layer > 0:
                net = tl.layers.DropoutLayer(net, keep=keep_prob,is_fix=True,
                                                 is_train=is_training, name='drop'+str(layer+1))
            if normalization == 'LN':
                net = tl.layers.LayerNormLayer(net, act=tf.identity, trainable=is_training, name='LN'+str(layer+1))
            if normalization == 'BN':
                net = tl.layers.BatchNormLayer(net, act=tf.identity, is_train=is_training, name='BN'+str(layer+1))

        logits = DenseLayer(net, n_units=num_classes, act=tf.identity, W_init=W_init2, name='fc'+str(nlayer))
        L2 = 0.0

    end_points['layer'] = logits.all_layers;
    end_points['param'] = logits.all_params;

    return logits.outputs, end_points, L2,

