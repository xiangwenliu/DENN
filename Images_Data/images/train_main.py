from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from nets import nets_factory
from preprocessing import preprocessing_factory


parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='~/tensorflowproject/data/mnistv',
                    help='The path to the CIFAR-10/CIFAR-100/mnist/mnist-fashion data directory.')

parser.add_argument('--train_dir', type=str, default='./log',
                    help='The directory where the model will be stored.')

parser.add_argument('--best_dir', type=str, default='./logbest',
                    help='The directory where the model will be stored.')

parser.add_argument('--dataset_name', type=str, default='mnist',
                    help='The dataset name.')

parser.add_argument('--model_name', type=str, default='fullnet',
                    help='The model name you choose.')

parser.add_argument('--train_epochs', type=int, default=100,
                    help='The number of epochs to train.')

parser.add_argument('--batch_size', type=int, default=100,
                    help='The number of images per batch.')

parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='The leatning rate.')

parser.add_argument('--normalization', type=str, default='NO',
                    help='The Normalization you choose, default Layer Normalization.')

parser.add_argument('--factor', type=float, default=0.1,
                    help='A multiplicative factor by which the values will be scaled.')

parser.add_argument('--l2_scale', type=float, default=0.0001,
                    help='The scale of L2 regularizer, 0.0 disables the L2 regularizer.')

parser.add_argument('--keep_probablity', type=float, default=0.9,
                    help='The keep probablity of dropout.')

parser.add_argument('--branches', type=int, default=2,
                    help='The branch number of dendritic neural network model.')

parser.add_argument('--layer_number', type=int, default=3,
                    help='The layer number of  network model.')

parser.add_argument('--net_length', type=int, default=512,
                    help='The unit number of  network model in every hidden layer.')

parser.add_argument('--branch_dir', type=str, default='./',
                    help='The directory where the information will be stored.')

FLAGS = parser.parse_args()

#dataset file name
TRAIN_FILE = 'train.tfrecord'
VALIDATION_FILE = 'validation.tfrecord'
TEST_FILE = 'test.tfrecord'


def read_data(path, shape, file, datasetname, batch_size=100,is_train=False):
    filename = os.path.join(path, file)
    # print('-------------------data_dir:' + filename)
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    if datasetname == 'mnist' or datasetname == 'fashion':
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        # img.set_shape(shape)
        img = tf.reshape(img, shape)
        img = tf.to_float(img)
        preprocessing_fn = preprocessing_factory.get_preprocessing(datasetname,is_training=is_train)
        img = preprocessing_fn(img, 28, 28)
    elif datasetname == 'cifar10' or datasetname == 'cifar100':
        img = tf.decode_raw(features['img_raw'], tf.float32)
        img = tf.reshape(img, shape)
        image = tf.to_float(img)
        img = tf.image.per_image_standardization(img)
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return img, label


def main(unused_argv):
    num_classes = 10
    train_num_sample = 50000
    validation_num_sample = 5000

    if FLAGS.dataset_name == 'cifar10':
        train_file = 'train.tfrecord'
        validation_file = 'validation.tfrecord'
        shape = [32, 32, 3]
        # shape = [32*32*3]
        num_classes = 10
        train_num_sample = 48000
        validation_num_sample = 2000
        test_num_sample = 10000
        model_file_name = "model_cifar10_tfrecord.ckpt"

    if FLAGS.dataset_name == 'cifar100':
        train_file = 'train.tfrecord'
        validation_file = 'validation.tfrecord'
        shape = [32, 32, 3]
        num_classes = 100
        train_num_sample = 48000
        validation_num_sample = 2000
        test_num_sample = 10000
        model_file_name = "model_cifar100_tfrecord.ckpt"

    if FLAGS.dataset_name == 'mnist':
        train_file = 'train.tfrecord'
        validation_file = 'validation.tfrecord'
        # shape = [28 * 28]
        shape = [28, 28, 1]
        num_classes = 10
        train_num_sample = 55000
        validation_num_sample = 5000
        test_num_sample = 10000
        model_file_name = "model_mnist_tfrecord.ckpt"
        optimizer = tf.train.GradientDescentOptimizer
    if FLAGS.dataset_name == 'fashion':
        train_file = 'train.tfrecord'
        validation_file = 'validation.tfrecord'
        # shape = [28 * 28]
        shape = [28, 28, 1]
        num_classes = 10
        train_num_sample = 55000
        validation_num_sample = 5000
        test_num_sample = 10000
        model_file_name = "model_fashion_tfrecord.ckpt"
        optimizer = tf.train.GradientDescentOptimizer
    with tf.device('/cpu'):
        tl.files.exists_or_mkdir(FLAGS.train_dir)

        x_train_, y_train_ = read_data(path=FLAGS.data_dir,shape=shape,file=TRAIN_FILE, datasetname=FLAGS.dataset_name, is_train=True)
        x_validate_, y_validate_ = read_data(path=FLAGS.data_dir, shape=shape, file=VALIDATION_FILE, datasetname=FLAGS.dataset_name, is_train=False)
        x_test_, y_test_ = read_data(path=FLAGS.data_dir, shape=shape, file=TEST_FILE, datasetname=FLAGS.dataset_name, is_train=False)

        batch_size = FLAGS.batch_size
        resume = False  # load model, resume from previous checkpoint?
        x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
            batch_size=batch_size, capacity=2000, min_after_dequeue=1000, num_threads=4) # set the number of threads here
        # for validation, uses batch instead of shuffle_batch
        x_validate_batch, y_validate_batch = tf.train.batch([x_validate_, y_validate_],
                                                    batch_size=batch_size, capacity=2000, num_threads=4)
        # for testing, uses batch instead of shuffle_batch
        x_test_batch, y_test_batch = tf.train.batch([x_test_, y_test_],
            batch_size=batch_size, capacity=2000, num_threads=4)

        with tf.device('/gpu'):
            # FLAGS.branches == -1 train FNN, FLAGS.branches == 0 train FNN with Batch Normalization
            # FLAGS.branches == 0 train FNN with Layer Normalization
            if FLAGS.branches == -1:
                modelname = 'fullnet'
                meshead = 'full'
            elif FLAGS.branches == 0:
                modelname = 'fullnet'
                meshead = 'BN-full'
                FLAGS.normalization = 'BN'
            elif FLAGS.branches == 1:
                modelname = 'fullnet'
                meshead = 'LN-full'
                FLAGS.normalization = 'LN'
            else:
                if FLAGS.model_name == 'dendritenet':
                    modelname = 'dendritenet'
                    meshead = 'dendrite' + str(FLAGS.branches)

                else:
                    modelname = 'maxoutnet'
                    meshead = 'maxoutnet' + str(FLAGS.branches)


            network_model = nets_factory.get_network_fn(modelname)
            with tf.variable_scope("model") as scope:
                logits, end_points, regu2 = network_model(x_train_batch, num_classes,
                                                                        is_training=True,
                                                                        factor=FLAGS.factor,
                                                                        l2_scales=FLAGS.l2_scale,
                                                                        nlayer=FLAGS.layer_number,
                                                                        branch=FLAGS.branches,
                                                                        netlen=FLAGS.net_length,
                                                                        keep_prob=FLAGS.keep_probablity,
                                                                        normalization=FLAGS.normalization)
                scope.reuse_variables()
                logits_validation, end_points_validation, validationregu2 = network_model(x_validate_batch, num_classes,
                                                                                            is_training=False,
                                                                                            factor=FLAGS.factor,
                                                                                            l2_scales=FLAGS.l2_scale,
                                                                                            nlayer=FLAGS.layer_number,
                                                                                            branch=FLAGS.branches,
                                                                                            netlen=FLAGS.net_length,
                                                                                            keep_prob=FLAGS.keep_probablity,
                                                                                            normalization=FLAGS.normalization)
                logits_test, end_points_test, testregu2 = network_model(x_test_batch, num_classes,
                                                                          is_training=False,
                                                                          factor=FLAGS.factor,
                                                                          l2_scales=FLAGS.l2_scale,
                                                                          nlayer=FLAGS.layer_number,
                                                                          branch=FLAGS.branches,
                                                                          netlen=FLAGS.net_length,
                                                                          keep_prob=FLAGS.keep_probablity,
                                                                          normalization=FLAGS.normalization)

        #training
        cost = tl.cost.cross_entropy(logits, y_train_batch, name='cost')
        cost = cost + regu2
        correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_train_batch)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #validation
        cost_val = tl.cost.cross_entropy(logits_validation, y_validate_batch, name='cost_val')
        cost_val = cost_val + validationregu2
        correct_prediction_val = tf.equal(tf.cast(tf.argmax(logits_validation, 1), tf.int32), y_validate_batch)
        acc_val = tf.reduce_mean(tf.cast(correct_prediction_val, tf.float32))
        #test
        cost_test = tl.cost.cross_entropy(logits_test, y_test_batch, name='cost_val')
        cost_test = cost_test + testregu2
        correct_prediction_test = tf.equal(tf.cast(tf.argmax(logits_test, 1), tf.int32), y_test_batch)
        acc_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))

        #training the model
        n_epoch = FLAGS.train_epochs
        print_freq = 1
        n_step_epoch = int(train_num_sample / batch_size)
        n_step = n_epoch * n_step_epoch
        train_acc_list = []
        test_acc_list = []
        train_loss_list = []
        test_loss_list = []
        val_acc_list = []
        val_loss_list = []
        max_val_acc = 0.0
        final_test_acc = 0.0
        final_test_loss = 0.0

        # exponential decay learning rate
        LR_start = FLAGS.learning_rate
        LR_fin = 1.0e-5
        LR_decay = (LR_fin / LR_start) ** (1.0 / n_epoch)
        step_decay = n_step_epoch

        global_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_steps, step_decay,LR_decay, staircase=False)
        # learning_rate = FLAGS.learning_rate
        with tf.device('/gpu'):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_steps)


        summaries = []
        summaries.append(tf.summary.scalar('learning_rate', learning_rate,collections=['learningrate']))
        # Merge summaries together.
        summary_op = tf.summary.merge(summaries, name='summary_op')


        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=1)
        print('   learning_rate: %f' % FLAGS.learning_rate)
        print('   batch_size: %d' % batch_size)
        print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))
        if resume:
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            if checkpoint_path is not None:
                saver.restore(sess, checkpoint_path)
                print('-------------Loading existing model %s' % checkpoint_path)
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        dt0 = time.time()

        # summary_str_variable = sess.run(summary_op_variable)
        # summary_writer.add_summary(summary_str_variable, 0)
        # summary_writer.flush()
        for epoch in range(n_epoch):
            sys.stdout.flush()
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, epoch+1)
            summary_writer.flush()

            start_time = time.time()
            train_loss, train_acc, n_batch = 0.0, 0.0, 0
            for s in range(n_step_epoch):
                err, ac, _ = sess.run([cost, acc, train_op])
                step += 1;
                train_loss += err;
                train_acc += ac;
                n_batch += 1

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d : Step %d-%d of %d took %fs" % (
                epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
                print("   train loss: %f" % (train_loss / n_batch))
                print("   train acc: %f" % (train_acc / n_batch))
                # write train accuracy and loss
                summary1 = tf.Summary(value=[
                    tf.Summary.Value(tag="train_loss", simple_value=train_loss / n_batch),
                    tf.Summary.Value(tag="train_acc", simple_value=train_acc / n_batch),
                ])
                summary_writer.add_summary(summary1, epoch+1)
                summary_writer.flush()
                train_acc_list.append(train_acc / n_batch)
                train_loss_list.append(train_loss / n_batch)

                #validation
                val_loss, val_acc, n_batch = 0.0, 0.0, 0
                val_batch = int(validation_num_sample / batch_size)
                for _ in range(val_batch):
                    err, ac = sess.run([cost_val, acc_val])
                    val_loss += err / val_batch;
                    val_acc += ac / val_batch;

                print("   validate loss: %f" % val_loss)
                print("   validate acc: %f" % val_acc)
                #validation accuracy and loss
                summary3 = tf.Summary(value=[
                    tf.Summary.Value(tag="validate_loss", simple_value=val_loss),
                    tf.Summary.Value(tag="validate_acc", simple_value=val_acc),
                ])
                summary_writer.add_summary(summary3, epoch + 1)
                summary_writer.flush()
                val_acc_list.append(val_acc)
                val_loss_list.append(val_loss)
                if val_acc > max_val_acc:
                    max_val_acc = val_acc


                    #test model
                    test_loss, test_acc, n_batch = 0.0, 0.0, 0
                    for _ in range(int(test_num_sample / batch_size)):
                        err, ac = sess.run([cost_test, acc_test])
                        test_loss += err;
                        test_acc += ac;
                        n_batch += 1
                    t1 = "   test loss: %f" % (test_loss / n_batch)
                    t2 = "   test acc: %f" % (test_acc / n_batch)
                    print(t1)
                    print(t2)
                    final_test_acc = test_acc / n_batch
                    final_test_loss = test_loss / n_batch
                    test_acc_list.append(test_acc / n_batch)
                    test_loss_list.append(test_loss / n_batch)

        dt1 = time.time()
        print('-------------------training total cost time: %f seconds' % (dt1 - dt0))
        #write accuracy and loss to file
        str1 = 'train_acc_list=%s' % (train_acc_list)
        str2 = 'train_loss_list=%s' % (train_loss_list)
        str3 = 'test_acc_list=%s' % (test_acc_list)
        str4 = 'test_loss_list=%s' % (test_loss_list)
        str5 = 'val_acc_list=%s' % val_acc_list
        str6 = 'val_loss_list=%s' % val_loss_list
        file = open(os.path.join(FLAGS.branch_dir, 'accuracylist.txt'), 'a')
        file.write(str1 + '\r\n')
        file.write(str2 + '\r\n')
        file.write(str3 + '\r\n')
        file.write(str4 + '\r\n')
        file.write(str5 + '\r\n')
        file.write(str6 + '\r\n')
        file.close()

        #--------------------------------------------
        mes1 = meshead+'=%s' % (train_acc_list)
        mes2 = meshead + '=%s,%s' % (max_val_acc, final_test_acc)
        trainloss = meshead + '=%s' % (train_loss_list)
        testloss = meshead + '=%s' % (final_test_loss)
        mes_val_acc = meshead+'=%s' % (val_acc_list)
        mes_val_loss = meshead + '=%s' % (val_loss_list)

        #============================write to txt file
        file = open('trainacclist.txt', 'a')
        file.write(mes1 + '\r\n')
        file.close()
        file = open('testacclist.txt', 'a')
        file.write(mes2 + '\r\n')
        file.close()
        file = open('trainloss.txt', 'a')
        file.write(trainloss + '\r\n')
        file.close()
        file = open('testloss.txt', 'a')
        file.write(testloss + '\r\n')
        file.close()
        file = open('validationacclist.txt', 'a')
        file.write(mes_val_acc + '\r\n')
        file.close()
        file = open('validationloss.txt', 'a')
        file.write(mes_val_loss + '\r\n')
        file.close()

        #=============================
        coord.request_stop()
        coord.join(threads)
        summary_writer.close()
        sess.close()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
