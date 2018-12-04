# coding: utf-8
# Training DENN on UCI datasets
import tensorflow as tf
import pandas as pd
import numpy as np
import random
from tensorlayer.layers import *
import tensorlayer as tl
import mylayer as mylayer
import os
import time
import sys
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--data_set', type=str, default='trains',
                    help='The  data set name.')

parser.add_argument('--model_name', type=str, default='dendritenet',
                    help='The model name you choose.')

parser.add_argument('--data_dir', type=str, default='./data/contrac/contrac_py.dat',
                    help='The path to the data set directory.')

parser.add_argument('--label_dir', type=str, default='./data/contrac/labels_py.dat',
                    help='The path to the data set directory.')

parser.add_argument('--test_folds', type=str, default='./data/contrac/folds_py.dat',
                    help='The path to the data set directory.')

parser.add_argument('--validation_folds', type=str, default='./data/contrac/validation_folds_py.dat',
                    help='The path to the data set directory.')

parser.add_argument('--result_dir', type=str, default='',
                    help='The directory where the result will be stored.')

parser.add_argument('--branches', type=int, default='4',
                    help='The branch number of the dendritic net model.')

parser.add_argument('--layers', type=int, default='3',
                    help='The layer number of the dendritic net model.')

parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='The leatning rate.')

parser.add_argument('--keep_prob', type=float, default=0.8,
                    help='The dropout rate.')

parser.add_argument('--net_length', type=int, default='512',
                    help='The units of eatch layer.')

parser.add_argument('--layer_form', type=str, default='rect',
                    help='The Layer Form of network.')

parser.add_argument('--Normalization', type=str, default='No',
                    help='Which normalization you select in your network.')


FLAGS = parser.parse_args()


#fix learning rate no summary
train_accuracy_file = 'trainacclist.txt'
train_loss_file = 'trainloss.txt'
test_accuracy_file = 'testacclist.txt'
test_loss_file = 'testloss.txt'
validation_acc_file = 'validationacclist.txt'
avg_validation_acc_file = 'avgvalidationacclist.txt'
fold_test_acc_file = 'foldtestacc.txt'
data_grad_search_file = FLAGS.data_set + '.txt'


train_dir = './log'
TEST_SIZE = 0.1  # Test set size (% of dataset)
VALIDATE_SIZE = 0.1
LEARNING_RATE = FLAGS.learning_rate  # Learning rate
LEARNING_RATE_END = 1.0e-5
TRAINING_EPOCHS = 100  # Number of epochs
batch_size = 128  # Batch size
DISPLAY_STEP = 1  # Display progress each x epochs
netlen = FLAGS.net_length  # Number of hidden neurons 256
STDDEV = 0.1  # Standard deviation (for weights random init)
RANDOM_STATE = 100  # Random state for train_test_split
oneper=0.8
Normalization = FLAGS.Normalization
WINDOW_LENGTH = 10
FOLD_TIME = 4
N_LAYER = FLAGS.layers
dropoutrate = 1.0 - FLAGS.keep_prob
EPS = 1.0e-7
print_freq = 1
smoothing = 10
span = 20


def read_data(filename):
    result = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            str = line.strip()
            if len(str) > 1:
                arrylist = str.split(',')
                result.append(arrylist)
    arr = np.array(result, dtype=np.float32)
    return arr


def shuffle_train_data(train_data, train_label):
    permutation = np.random.permutation(train_label.shape[0])
    train_data = train_data[permutation, :]
    train_label = train_label[permutation]
    return train_data,train_label


def dendritnet(images,branch=2,num_classes=10, is_training=False,
               reuse=True,
               keep_prob=0.8,
               scope='DendritNet'):
    # W_init = tf.truncated_normal_initializer(stddev=5e-2)
    W_init2 = tf.contrib.layers.variance_scaling_initializer(factor=2.0)
    act_func = tf.identity
    if Normalization == 'No':
        act_func = tf.nn.relu
    with tf.variable_scope(scope, 'DendritNet', [images, num_classes]):
        tl.layers.set_name_reuse(reuse)
        net = InputLayer(images, name='input')

        for layer in range(N_LAYER - 1):
            n_in = int(net.outputs.get_shape()[-1])

            if layer == 0:
                # the first layer use Denselayer
                net = DenseLayer(net, n_units=netlen, act=act_func, W_init=W_init2, name='fc' + str(layer + 1))
            if layer != 0:
                net = mylayer.MyDenseLayer(net, netlen, act=tf.identity,W_init=W_init2, branch=branch,keep_prob=keep_prob,
                                       is_scale=False, is_train=is_training, name='dfc'+str(layer+1))

        logits = DenseLayer(net, n_units=num_classes, act=tf.identity, W_init=W_init2, name='fc'+str(N_LAYER))
    return logits


def get_smooth_mean(arr, kernal_size):
    kernal = np.ones(int(kernal_size))/float(kernal_size)
    return np.convolve(arr, kernal, 'same')


if __name__ == '__main__':
    #load data
    data_file = FLAGS.data_dir
    label_file = FLAGS.label_dir
    data = pd.read_csv(data_file,header=None)
    label = pd.read_csv(label_file, header=None)

    validation_remark = pd.read_csv(FLAGS.validation_folds, header=None)
    test_remark = pd.read_csv(FLAGS.test_folds, header=None)

    # split training, validation and test data
    train_index = (test_remark.values[:, 0] == 0)
    test_index = (test_remark.values[:, 0] == 1)
    val = (validation_remark.values[:, 0] == 1)
    validation_index = (train_index & (val == 1))
    train_index = (train_index & (val == 0))

    train_x = data.as_matrix()[train_index, :]
    train_y = label.as_matrix()[train_index, 0]
    train_y[np.isnan(train_y)] = -1

    test_x = data.as_matrix()[test_index, :]
    test_y = label.as_matrix()[test_index, 0]
    test_y[np.isnan(test_y)] = -1

    validation_x = data.as_matrix()[validation_index, :]
    validation_y = label.as_matrix()[validation_index, 0]
    validation_y[np.isnan(validation_y)] = -1

    N_INPUT = train_x.shape[1]  # data input
    N_CLASSES = int(np.max(train_y) + 1)  # total classes

    print('N_INPUT=', N_INPUT)
    print('N_CLASSES=', N_CLASSES)
    print("Data loaded and splitted successfully...\n")
    # scaler = preprocessing.StandardScaler().fit(train_x)
    # validation_x = scaler.transform(validation_x)
    # test_x = scaler.transform(test_x)
    #
    # print('*******************************************DACAY_STEP=',DACAY_STEP)
    start_time = time.time()


    # Launch session
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # the number of input features and output classes
        n_input = N_INPUT  # input features
        n_classes = N_CLASSES  # output  classes
        # placeholders variables
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("int64", [None])
        is_training = tf.placeholder(tf.bool)
        dropoutRate = tf.placeholder(tf.float32)
        # build network model
        network = dendritnet(x,branch=FLAGS.branches, num_classes=N_CLASSES, is_training=is_training,keep_prob=FLAGS.keep_prob,
                          scope='dendrite'+str(FLAGS.branches))
        pred = network.outputs

        # loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y,
                                                                       name='softmax_cross_entropy')
        out_gradient = tf.stop_gradient(tf.cast(tf.not_equal(y, -1), tf.float32))
        cross_entropy *= out_gradient
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

        # accuracy
        prediction = tf.equal(y, tf.argmax(pred, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        print("Network Model built successfully...\n")
        print("--------------------------Starting training the models...\n")

        avg_test_acc = 0.0
        avg_validation_acc = 0.0
        fold_test_acc_list = []

        val_acc_list = np.zeros([TRAINING_EPOCHS])
        train_acc_list = np.zeros([TRAINING_EPOCHS])
        test_acc_list = np.zeros([TRAINING_EPOCHS])
        early_stop_step = 0
        val_best = 0

        # batch size
        batch_size = int(min(batch_size, train_x.shape[0] / 5))
        if batch_size < 1:
            batch_size = train_x.shape[0]
        print('=============================batch size=', batch_size)
        # summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        max_val_acc = 0.0
        max_fold_test_acc = 0.0
        flag = False
        total_batch = train_x.shape[0] // batch_size
        # Training loop
        for epoch in range(TRAINING_EPOCHS):
            sys.stdout.flush()

            if ((epoch <= (early_stop_step + span))):
                # shuffle training data
                avg_train_loss = 0.
                avg_train_accuracy = 0.
                indexs = np.random.permutation(np.arange(train_x.shape[0]))
                for i in range(total_batch):
                    train_batch_x = train_x[indexs[i * batch_size:(i + 1) * batch_size]]
                    train_batch_y = train_y[indexs[i * batch_size:(i + 1) * batch_size]]
                    # train model
                    _,train_acc,train_loss = sess.run([optimizer,accuracy, loss],
                                                      feed_dict={x: train_batch_x, y: train_batch_y,
                                                                 dropoutRate: dropoutrate,is_training: True})
                    avg_train_loss += train_loss / total_batch
                    avg_train_accuracy += train_acc / total_batch
                # show logs
                if epoch % print_freq == 0:
                    ### Validation accuracy and loss
                    val_acc, val_loss = sess.run([accuracy, loss],
                                                 feed_dict={x: validation_x, y: validation_y,
                                                            dropoutRate: 0.0, is_training: False})
                    val_acc_list[epoch] = val_acc
                    if (val_acc > val_best):
                        val_best = val_acc
                        early_stop_step = epoch

                    ### Test accuracy and loss
                    test_acc, test_loss = sess.run([accuracy, loss],
                                                   feed_dict={x: test_x, y: test_y,
                                                              dropoutRate: 0.0,is_training: False})
                    test_acc_list[epoch] = test_acc
                    print("Epoch %d :" % epoch)
                    print("  train loss: %f" % (avg_train_loss))
                    print("  train acc: %f" % (avg_train_accuracy))
                    print("  validate loss: %f" % val_loss)
                    print("  validate acc: %f" % val_acc)
                    print("  test loss: %f" % test_loss)
                    print("  test acc: %f" % test_acc)
        if (np.any(np.isnan(val_acc_list))):
            max_val_acc = 0.0
            max_fold_test_acc = 0.0
        else:
            smooth_mean = get_smooth_mean(val_acc_list, smoothing)
            res = test_acc_list[smooth_mean.argmax()]
            print("The accuracy on test set: %f" % res)
            max_val_acc = smooth_mean.max()
            max_fold_test_acc = res

        str_res = ''
        if FLAGS.branches == 0:
            str_res = ('%s,%d,%d,%f,%f,%s,%f,%f' %(FLAGS.data_set, N_LAYER, netlen, LEARNING_RATE, dropoutrate,
                                               FLAGS.layer_form, max_val_acc, max_fold_test_acc))
        if FLAGS.branches != 0:
            str_res = ('%s,%d,%d,%d,%f,%f,%f,%f' % (FLAGS.data_set, FLAGS.branches, N_LAYER, netlen, LEARNING_RATE,FLAGS.keep_prob,
                                              max_val_acc, max_fold_test_acc))
        # write to file
        file = open(os.path.join(FLAGS.result_dir, data_grad_search_file), 'a')
        file.write(str_res + '\r\n')
        file.close()

        print("-----------End training model cost time %f seconds.\n" % ( time.time() - start_time))
