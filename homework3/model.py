import tensorflow as tf
import atari_wrappers  # from OpenAI Baselines
from collections import deque


class Model:
    def training_op(self):
        cost = tf.reduce_mean(tf.square(self.y - self.q_value))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(cost, global_step=self.global_step)

    def dqn_network(self, name):
        # specify the network, none is for dynamic
        with tf.variable_scope(name) as scope:
            hidden_1 = tf.layers.conv2d(self.x, 32, (8, 8), (3, 3), padding='same', activation=tf.nn.relu, name='hidden_1')
            hidden_2 = tf.layers.conv2d(hidden_1, 64, (4, 4), (2, 2), padding='same', activation=tf.nn.relu,
                                        name='hidden_2')
            hidden_3 = tf.layers.conv2d(hidden_2, 64, (3, 3), (1, 1), padding='same', activation=tf.nn.relu,
                                        name='hidden_3')
            flatten = tf.reshape(hidden_3, shape=[-1, 64 * 14 * 14], name='flatten')
            hidden_4 = tf.layers.dense(flatten, 512, activation=tf.nn.relu,
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       name='hidden_4')
            output = tf.layers.dense(hidden_4, self.n_outputs, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     name='output')

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = {var.name[len(scope.name):]: var
                                  for var in trainable_vars}
        tf.identity(output, name='output')
        return output, trainable_vars_by_name


    def __init__(self, params):
        self.env = atari_wrappers.wrap_deepmind(atari_wrappers.make_atari('SeaquestNoFrameskip-v4'), frame_stack=True)
        self.replay_memory_size = params['replay_memory'] if 'replay_memory' in params else 10000
        self.replay_memory = deque([], maxlen=self.replay_memory_size)
        self.n_steps = params['n_steps'] if 'n_steps' in params else 100000
        self.training_start = params['training_start'] if 'training_start' in params else 1000
        self.training_interval = params['training_interval'] if 'training_interval' in params else 3
        self.save_steps = params['save_steps'] if 'save_steps' in params else 50
        self.copy_steps = params['copy_steps'] if 'copy_steps' in params else 25
        self.discount_rate = params['discount_rate'] if 'discount_rate' in params else 0.95
        self.skip_start = params['skip_start'] if 'skip_start' in params else 90
        self.batch_size = params['batch_size'] if 'batch_size' in params else 50
        self.iteration = params['iteration'] if 'iteration' in params else 0
        self.n_outputs = params['n_outputs'] if 'n_outputs' in params else self.env.action_space.n
        self.learning_rate = params['learning_rate'] if 'learning_rate' in params else 0.001
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.x = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name="input_placeholder")
        self.x_action = tf.placeholder(tf.int32, shape=[None], name="x_action")
        self.y = tf.placeholder(tf.float32, [None, 1])

        # setup models, replay memory, and optimizer
        self.actor_q_values, actor_vars = self.dqn_network("q_network/actor")
        critic_q_values, self.critic_vars = self.dqn_network("q_network/critic")
        self.q_value = tf.reduce_sum(critic_q_values * tf.one_hot(self.x_action, self.n_outputs),
                                     axis=1, keep_dims=True)
        copy_ops = [actor_var.assign(self.critic_vars[var_name])
                    for var_name, actor_var in actor_vars.items()]
        self.copy_critic_to_actor = tf.group(*copy_ops)
        self.train_op = self.training_op()
