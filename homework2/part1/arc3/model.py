# TensorFlow code that defines the network
import argparse
import os
import tensorflow as tf
import util
import numpy as np

parser = argparse.ArgumentParser(description='Classify  CIFAR-100 images.')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/work/cse496dl/shared/homework/02',
    help='directory where CIFAR-100 is located')
parser.add_argument(
    '--model_dir',
    type=str,
    default='./CIPHAR_100_logs',
    help='directory where model graph and weights are saved')

class Model:
    def __init__(self, params):
        self.args = parser.parse_args()
        self.batch_size = params['batch_size'] if 'batch_size' in params else 25
        self.epochs = params['epochs'] if 'epochs' in params else 1000
        self.test_set_size = params['test_set_size'] if 'test_set_size' in params else .3
        self.validation_set_size = params['validation_set_size'] if 'validation_set_size' in params else .3
        self.early_stopping = params['early_stopping'] if 'early_stopping' in params else 5
        self.activation = params['activation'] if 'activation' in params else tf.nn.relu
        reg_scale = params['reg_scale'] if 'reg_scale' in params else 0.001
        self.regularizer = params['regularizer'] if 'regularizer' in params else tf.contrib.layers.l2_regularizer(reg_scale)
        data = self.my_data()
        self.train_images = data['train_images']
        self.train_labels = data['train_labels']
        self.test_images = data['test_images']
        self.test_labels = data['test_labels']
        self.validation_images = data['validation_images']
        self.validation_labels = data['validation_labels']
        self.x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_placeholder')
        self.y = tf.placeholder(tf.float32, [None, 100], name='label')
        self.global_step_tensor = tf.get_variable(
            'global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
        reduction_rate = params['reduction_rate'] if 'reduction_rate' in params else .96
        starter_learning_rate = params['starter_learning_rate'] if 'starter_learning_rate' in params else .1
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step_tensor,
                                           data['train_images'].shape[0], reduction_rate, staircase=True)
        self.output = self.network()

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.y, logits=self.output)
        self.saver = tf.train.Saver(max_to_keep=self.early_stopping)

    def my_data(self):
        # load training data
        train_images = util.np.load(os.path.join(self.args.data_dir, 'cifar_images.npy'))

        # normalize data
        train_images = train_images / 255

        # reshape to fit input tensor
        train_images = np.reshape(train_images, [-1, 32, 32, 3]) # `-1` means "everything not otherwise accounted for"

        # load training labels
        train_labels = util.np.load(os.path.join(self.args.data_dir, 'cifar_labels.npy'))
        train_images, train_labels = util.shuffler(train_images, train_labels)

        # convert labels to one-hots
        train_labels = tf.Session().run(tf.one_hot(train_labels, 100))

        # set up test set
        test_images, train_images = util.split_data(train_images, self.test_set_size)
        test_labels, train_labels = util.split_data(train_labels, self.test_set_size)

        # set up validation set
        validation_images, train_images = util.split_data(train_images, self.validation_set_size)
        validation_labels, train_labels = util.split_data(train_labels, self.validation_set_size)

        data = {
            "train_images": train_images,
            "train_labels": train_labels,
            "test_images": test_images,
            "test_labels": test_labels,
            "validation_images": validation_images,
            "validation_labels": validation_labels
        }

        return data

    def network(self):
        with tf.name_scope('linear_model') as scope:
            conv_1 = tf.layers.conv2d(inputs=self.x, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv_1')
            pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2, padding='same', name='pool_1')
            conv_2 = tf.layers.conv2d(pool_1, filters=128, kernel_size=1, padding='same', activation=tf.nn.relu, name='conv_2')
            pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2, padding='same', name='pool_2')
            flatten_dim = np.prod(pool_2.get_shape().as_list()[1:])
            flat = tf.reshape(pool_2, [-1, flatten_dim])
            fully_connected = tf.layers.dense(flat, 400, activation=self.activation, kernel_regularizer=self.regularizer, name='fully_connected')
            output_layer = tf.layers.dense(fully_connected, 100, name='output_layer')
        tf.identity(output_layer, name='output')
        return output_layer

    def confusion_matrix_op(self):
        return tf.confusion_matrix(
            tf.argmax(self.y, axis=1),
            tf.argmax(self.output, axis=1), num_classes=100)

    def train_op(self):
        # set up training and saving functionality
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        training_op = optimizer.minimize(self.cross_entropy, global_step=self.global_step_tensor)

        return training_op
