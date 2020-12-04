# TensorFlow code that defines the network
import argparse
import os
import tensorflow as tf
import util

parser = argparse.ArgumentParser(description='Classify  Fashion-MNIST images.')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/work/cse496dl/shared/homework/01',
    help='directory where fashion-MNIST is located')
parser.add_argument(
    '--model_dir',
    type=str,
    default='./Fashion_MNIST_logs',
    help='directory where model graph and weights are saved')

class Model:
    def __init__(self, params):
        self.args = parser.parse_args()
        self.batch_size = params['batch_size'] if 'batch_size' in params else 256
        self.epochs = params['epochs'] if 'epochs' in params else 100
        self.test_set_size = params['test_set_size'] if 'test_set_size' in params else .1
        self.validation_set_size = params['validation_set_size'] if 'validation_set_size' in params else .3
        self.early_stopping = params['early_stopping'] if 'early_stopping' in params else 5
        self.activation = params['activation'] if 'activation' in params else tf.nn.relu
        self.regularizer = params['regularizer'] if 'regularizer' in params else tf.contrib.layers.l2_regularizer(0.001)
        data = self.my_data()
        self.train_images = data['train_images']
        self.train_labels = data['train_labels']
        self.test_images = data['test_images']
        self.test_labels = data['test_labels']
        self.validation_images = data['validation_images']
        self.validation_labels = data['validation_labels']
        self.x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
        self.y = tf.placeholder(tf.float32, [None, 10], name='label')
        self.global_step_tensor = tf.get_variable(
            'global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
        self.output = self.network()
        self.cross_entropy =  tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y, logits=self.output)
        self.saver = tf.train.Saver()

    def my_data(self):
        # load training data
        train_images = util.np.load(os.path.join(self.args.data_dir, 'fmnist_train_data.npy'))
        train_labels = util.np.load(os.path.join(self.args.data_dir, 'fmnist_train_labels.npy'))
        train_labels = tf.Session().run(tf.one_hot(train_labels, 10))
        
        # normalize data
        train_images = train_images / 255
        
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
            hidden = tf.layers.dense(
                self.x, 400, activation=self.activation, kernel_regularizer=self.regularizer, name='hidden_layer')
            output = tf.layers.dense(hidden, 10, name='output_layer')
        tf.identity(output, name='output')
        return output

    def confusion_matrix_op(self):
        # define classification loss
        return tf.confusion_matrix(
            tf.argmax(self.y, axis=1), tf.argmax(self.output, axis=1), num_classes=10)

    def train_op(self):
        # set up training and saving functionality
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(self.cross_entropy, global_step=self.global_step_tensor)

        return training_op

