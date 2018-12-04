import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import time
# from PIL imprt Image
import os
import io
import myfilecifar100 as myfile

data_path='./data/cifar100/'
X_train, y_train, X_test, y_test = myfile.load_cifar100_dataset(
                                    shape=(-1, 32, 32, 3),path=data_path, plotable=False)


def data_to_tfrecord(images, labels, filename):
    """ Save data into TFRecord """
    if os.path.isfile(filename):
        print("%s exists" % filename)
        return
    print("Converting data into %s ..." % filename)
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)
    writer2 = tf.python_io.TFRecordWriter(filename)
    print('-------------------image len is:%d'%(len(images)))

    for index, img in enumerate(images):
        img_raw = img.tobytes()
        ## Visualize a image
        # tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        label = int(labels[index])
        # print(label)
        ## Convert the bytes back to image as follow:
            # image = Image.frombytes('RGB', (32, 32), img_raw)
        # image = np.fromstring(img_raw, np.float32)
        # image = image.reshape([32, 32, 3])
        # tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()

def data_to_tfrecord_train_validate(images, labels, filename1, filename2):
    """ Save data into TFRecord """
    if os.path.isfile(filename1):
        print("%s exists" % filename1)
        return
    print("Converting data into %s ..." % filename1)
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename1)
    writer2 = tf.python_io.TFRecordWriter(filename2)
    print('-------------------image len is:%d'%(len(images)))
    per = 0
    labels.flags.writeable = True
    np.random.shuffle(labels)
    for index, img in enumerate(images):
        img_raw = img.tobytes()
        ## Visualize a image
        # tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        label = int(labels[index])
        # print(label)
        ## Convert the bytes back to image as follow:
            # image = Image.frombytes('RGB', (32, 32), img_raw)
        # image = np.fromstring(img_raw, np.float32)
        # image = image.reshape([32, 32, 3])
        # tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        if per < len(images) - 2000:
            writer.write(example.SerializeToString())  # Serialize To String
        else:
            writer2.write(example.SerializeToString())  # Serialize To String
        per += 1
    writer.close()
    writer2.close()

data_to_tfrecord_train_validate(images=X_train, labels=y_train, filename1=os.path.join(data_path, "train.tfrecord"),
                                filename2=os.path.join(data_path, "validation.tfrecord"))
data_to_tfrecord(images=X_test, labels=y_test, filename=os.path.join(data_path, "test.tfrecord"))
print('----------success!')