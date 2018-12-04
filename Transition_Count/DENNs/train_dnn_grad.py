import tensorflow as tf
import tensorlayer as tl
import numpy as np
import argparse
import sys
import random
from tensorlayer.layers import *
from tensorflow.examples.tutorials.mnist import input_data
from mylayer import MyDenseLayer

# dataset path
data_path='./data'

def model_feedforward(input_x, width, depth, branch, scale):
  inputs = InputLayer(input_x, name='input_layer')
  h1_value = []
  for i in range(depth):
    n_input = inputs.outputs.get_shape()[-1]
    inputs = MyDenseLayer(inputs, n_units=width, branch=branch,
                          W_init=tf.random_normal_initializer(
                            stddev=0.01, seed=1234),
                          name='dense_layer_' + str(i))
    h1_value.append(inputs.output_h1)
  return h1_value


def trajectory_circular(x, y, t):
  return np.cos(np.pi * t / 2.0) * x + np.sin(np.pi * t / 2.0) * y


def cal(width, depth, scale, branch, acc, times, eps, batchsize):
  with tf.device('/gpu'):
    print("#", width, depth, scale, branch, acc, times, eps, batchsize)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    log_dir = './log'
    tl.files.exists_or_mkdir(log_dir)
    mnist = input_data.read_data_sets \
      (data_path, one_hot=True)
    mnist_train = mnist.train.images
    data_x = tf.placeholder(tf.float32, shape=[batchsize, 784])
    cnt = np.zeros(depth)
    
    h1_value = model_feedforward(data_x, width, depth, branch, scale)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    random_list = [x for x in range(55000)]
    for i in range(times):
      num = np.reshape(np.array(random.sample(random_list, 2 * batchsize)),
                       [2, batchsize])
      mnist_x1 = np.reshape(mnist_train[num[0]], [batchsize, 784])
      mnist_x2 = np.reshape(mnist_train[num[1]], [batchsize, 784])
      for j in range(acc):
        t = (1.0 * j) / acc
        if j == 0:
          h1_value_pre = sess.run(h1_value,
                                  feed_dict={
                                    data_x: trajectory_circular(mnist_x1,
                                                                mnist_x2, t)})
          h1_value_pre = np.array(h1_value_pre)
          h1_value_pre = h1_value_pre.reshape([depth, batchsize, branch, width])
          index_pre = np.argmax(h1_value_pre, axis=2)
        else:
          h1_value_now = sess.run(h1_value,
                                  feed_dict={
                                    data_x: trajectory_circular(mnist_x1,
                                                                mnist_x2, t)})
          h1_value_now = np.array(h1_value_now)
          h1_value_now = h1_value_now.reshape([depth, batchsize, branch, width])
          index_now = np.argmax(h1_value_now, axis=2)
          
          cnt += np.average(np.sum(index_pre != index_now, axis=2),
                            axis=1)
          index_pre = index_now
        
        if j % 1000 == 0:
          print("Times %d, Step %d :" % (i, j))
          print(cnt)
          sys.stdout.flush()
      print("Times %d" % i)
      print(cnt)
      sys.stdout.flush()
    print("Final Result!: %d" % depth)
    cnt /= times
    print(cnt)
    print(np.sum(cnt))
    sys.stdout.flush()
    tf.reset_default_graph()
    sess.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='manual to this script')
  parser.add_argument('--width', type=int, default=512)
  parser.add_argument('--depth', type=int, default=2)
  parser.add_argument('--branch', type=int, default=2)
  parser.add_argument('--scale', type=int, default=10)
  parser.add_argument('--acc', type=int, default=100000)
  parser.add_argument('--times', type=int, default=5)
  parser.add_argument('--eps', type=float, default=1e-6)
  parser.add_argument('--batchsize', type=int, default=16)
  args = parser.parse_args()
  
  cal(width=args.width, depth=args.depth, scale=args.scale, branch=args.branch,
      acc=args.acc, times=args.times, eps=args.eps, batchsize=args.batchsize)

