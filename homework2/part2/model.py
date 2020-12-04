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
parser.add_argument(
    '--auto_encoder_dir',
    type=str,
    default='./ae/IMAGENET_LOGS',
    help='directory where auto encoder graph and weights are saved')

class AutoEncoder:
    def __init__(self):
        self.args = parser.parse_args()
        self.sparsity_weight = 5e-3
        self.batch_size = 1024
        self.imagenet_images = util.np.load(os.path.join(self.args.data_dir, 'imagenet_images.npy'))
        self.x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='autoencoder_input_placeholder')
        self.code = tf.placeholder(tf.float32, [None, 4, 4, 3], name='code_placeholder')
        self.global_step_tensor = tf.get_variable(
            'ae_global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
        [self.code, self.ae_output] = self.autoencoder_network(self.x)
                    
        self.saver = tf.train.Saver()
    
    def autoencoder_train_op(self):
        # calculate loss
        sparsity_loss = tf.norm(self.code, ord=1, axis=1)
        reconstruction_loss = tf.reduce_mean(tf.square(self.ae_output - self.x)) # Mean Square Error
        total_loss = reconstruction_loss + self.sparsity_weight * sparsity_loss

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(total_loss)

        return train_op
    
class Model:
    def __init__(self, params):
        # set hyperparameters
        self.args = parser.parse_args()
        self.batch_size = params['batch_size'] if 'batch_size' in params else 1024
        self.epochs = params['epochs'] if 'epochs' in params else 100
        self.test_set_size = params['test_set_size'] if 'test_set_size' in params else .1
        self.validation_set_size = params['validation_set_size'] if 'validation_set_size' in params else .3
        self.early_stopping = params['early_stopping'] if 'early_stopping' in params else 5
        self.activation = params['activation'] if 'activation' in params else tf.nn.relu
        reg_scale = params['reg_scale'] if 'reg_scale' in params else 0.001
        self.regularizer = params['regularizer'] if 'regularizer' in params else tf.contrib.layers.l2_regularizer(
            reg_scale)
        data = self.my_data()
        self.train_images = data['train_images']
        self.train_labels = data['train_labels']
        self.test_images = data['test_images']
        self.test_labels = data['test_labels']
        self.validation_images = data['validation_images']
        self.validation_labels = data['validation_labels']
        self.ae_path_prefix = params['path_prefix']
        self.ae_x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='ae_input_placeholder')
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

        # convert labels to one-hots
        train_labels = tf.Session().run(tf.one_hot(train_labels, 100))
        train_images, train_labels = util.shuffler(train_images, train_labels)

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

    def network(self, code_size=100):
        sess = tf.Session()
        saver = tf.train.import_meta_graph(self.ae_path_prefix + '.meta')
        saver.restore(sess, self.ae_path_prefix)
        graph = sess.graph

        self.ae_x = graph.get_tensor_by_name('autoencoder_input_placeholder:0')
        encoder = tf.stop_gradient(graph.get_tensor_by_name('encoder3/Conv2D:0'))

        # specify the network, none is for dynamic
        with tf.name_scope('linear_model') as scope:
            # A simple conv network with pooling
            # let's specify a conv stack
            conv_1 = tf.layers.conv2d(inputs=encoder, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv_1')
            pool_1 = tf.layers.max_pooling2d(conv_1, 2, 2, padding='same', name='pool_1')
            conv_2 = tf.layers.conv2d(pool_1, filters=128, kernel_size=1, padding='same', activation=tf.nn.relu, name='conv_2')
            pool_2 = tf.layers.max_pooling2d(conv_2, 2, 2, padding='same', name='pool_2')
            flatten_dim = np.prod(pool_2.get_shape().as_list()[1:])
            flat = tf.reshape(pool_2, [-1, flatten_dim])
            fully_connected = tf.layers.dense(flat, 400, activation=self.activation, kernel_regularizer=self.regularizer, name='fully_connected')
            output = tf.layers.dense(fully_connected, 100, name='output_layer')
        tf.identity(output, name='output')
        return output

    def confusion_matrix_op(self):
        return tf.confusion_matrix(
            tf.argmax(self.y, axis=1),
            tf.argmax(self.output, axis=1), num_classes=100)

    def network_train_op(self):
        # set up training and saving functionality
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(self.cross_entropy, global_step=self.global_step_tensor)

        return training_op
