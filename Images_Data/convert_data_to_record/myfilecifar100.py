#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import os
import numpy as np
import re
import sys
import tarfile
import gzip
import zipfile
import pickle
from six.moves import urllib
from six.moves import cPickle
from six.moves import zip
from tensorflow.python.platform import gfile
from tensorlayer.files.utils import maybe_download_and_extract


def load_cifar100_dataset(shape=(-1, 32, 32, 3), path='./data/cifar100', plotable=False, second=3):

    print("Load or Download cifar100 > {}".format(path))

    #Helper function to unpickle the data
    def unpickle(file):
        fp = open(file, 'rb')
        if sys.version_info.major == 2:
            data = pickle.load(fp)
        elif sys.version_info.major == 3:
            data = pickle.load(fp, encoding='latin-1')
        fp.close()
        return data

    # http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    filename = 'cifar-100-python.tar.gz'
    url = 'https://www.cs.toronto.edu/~kriz/'
    # #Download and uncompress file
    maybe_download_and_extract(filename, path, url, extract=True)

    #Unpickle file and fill in data
    X_train = None
    y_train = []
    data_dic = unpickle(os.path.join(path,'cifar-100-python/', 'train'))
    X_train = data_dic['data']
    y_train = data_dic['fine_labels']

    test_data_dic = unpickle(os.path.join(path,'cifar-100-python',  'test'))
    X_test = test_data_dic['data']
    y_test = np.array(test_data_dic['fine_labels'])

    if shape == (-1, 3, 32, 32):
        X_test = X_test.reshape(shape)
        X_train = X_train.reshape(shape)
    elif shape == (-1, 32, 32, 3):
        X_test = X_test.reshape(shape, order='F')
        X_train = X_train.reshape(shape, order='F')
        X_test = np.transpose(X_test, (0, 2, 1, 3))
        X_train = np.transpose(X_train, (0, 2, 1, 3))
    else:
        X_test = X_test.reshape(shape)
        X_train = X_train.reshape(shape)

    y_train = np.array(y_train)

    if plotable == True:
        print('\nCIFAR-100')
        import matplotlib.pyplot as plt
        fig = plt.figure(1)

        print('Shape of a training image: X_train[0]',X_train[0].shape)

        plt.ion()       # interactive mode
        count = 1
        for row in range(10):
            for col in range(10):
                a = fig.add_subplot(10, 10, count)
                if shape == (-1, 3, 32, 32):
                    plt.imshow(np.transpose(X_train[count-1], (1, 2, 0)), interpolation='nearest')
                elif shape == (-1, 32, 32, 3):
                    plt.imshow(X_train[count-1], interpolation='nearest')
                else:
                    raise Exception("Do not support the given 'shape' to plot the image examples")
                plt.gca().xaxis.set_major_locator(plt.NullLocator())    # 不显示刻度(tick)
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                count = count + 1
        plt.draw()      # interactive mode
        plt.pause(3)   # interactive mode

        print("X_train:",X_train.shape)
        print("y_train:",y_train.shape)
        print("X_test:",X_test.shape)
        print("y_test:",y_test.shape)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)

    return X_train, y_train, X_test, y_test

