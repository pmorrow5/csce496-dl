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
        self.batch_size = params['batch_size'] if 'batch_size' in params else 1024
        self.epochs = params['epochs'] if 'epochs' in params else 100
        self.test_set_size = params['test_set_size'] if 'test_set_size' in params else .1
        self.validation_set_size = params['validation_set_size'] if 'validation_set_size' in params else .3
        self.early_stopping = params['early_stopping'] if 'early_stopping' in params else 15
        self.activation = params['activation'] if 'activation' in params else tf.nn.relu
        self.regularizer = params['regularizer'] if 'regularizer' in params else tf.contrib.layers.l2_regularizer(0.001)
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
        self.output = self.network()

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.y, logits=self.output)
        self.saver = tf.train.Saver()

    def my_data(self):
        # load training data
        train_images = util.np.load(os.path.join(self.args.data_dir, 'cifar_images.npy'))

        # normalize data
        train_images = train_images / 255

        # reshape to fit input tensor
        train_images = np.reshape(train_images, [-1, 32, 32, 3]) # `-1` means "everything not otherwise accounted for"

        # load training labels
        train_labels = util.np.load(os.path.join(self.args.data_dir, 'cifar_labels.npy'))

        # convert labels to one-hots
        train_labels = tf.Session().run(tf.one_hot(train_labels, 100))

        # set up test set
        test_images, train_images = util.split_data(train_images, self.test_set_size)
        test_labels, train_labels = util.split_data(train_labels, self.test_set_size)

        # set up validation set
        train_images, train_labels = util.shuffler(train_images, train_labels)
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
        # specify the network, none is for dynamic
        with tf.name_scope('linear_model') as scope:

            # A simple conv network with pooling
            # let's specify a conv stack
            hidden_1 = tf.layers.conv2d(self.x, 32, 5, padding='same', activation=self.activation, name='hidden_1')
            pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same')
            hidden_2 = tf.layers.conv2d(pool_1, 64, 5, padding='same', activation=self.activation, name='hidden_2')
            pool_2 = tf.layers.max_pooling2d(hidden_2, 2, 2, padding='same')
            flatten_dim = np.prod(pool_2.get_shape().as_list()[1:])
            flat = tf.reshape(pool_2, [-1, flatten_dim])
            # followed by a connected layer
            c_layer1 = tf.layers.dense(flat, 4096, name='connected_layer1')
            c_layer2 = tf.layers.dense(c_layer1, 4096, name='connected_layer2')
            output = tf.layers.dense(c_layer2, 100, name='output')

        tf.identity(output, name='output')
        return output

    def confusion_matrix_op(self):
        # define classification loss

        return tf.confusion_matrix(
            tf.argmax(self.y, axis=1),
            tf.argmax(self.output, axis=1), num_classes=100)

    def train_op(self):
        # set up training and saving functionality
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(self.cross_entropy, global_step=self.global_step_tensor)

        return training_op
