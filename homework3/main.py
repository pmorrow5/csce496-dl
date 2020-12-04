# Code to specify and train the TensorFlow graph
import util
import model
import os
import tensorflow as tf
import numpy as np

checkpoint_path = "homework_3"
model_dir = "./SEAQUEST_LOGS"

training_starts = [500, 1000]
n_steps = [1000, 5000, 10000]

for training_start in training_starts:
    for n_step in n_steps:
        params = {
            "n_steps": n_step,
            "training_start": training_start,
        }

        m = model.Model(params)

        done = True  # env needs to be reset

        training_op = m.train_op
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        current_high_score = 0
        high_score = False

        with tf.Session() as sess:
            if os.path.isfile(os.path.join(model_dir, checkpoint_path)):
                saver.restore(sess, os.path.join(model_dir, checkpoint_path))
            else:
                init.run()
            while True:
                step = m.global_step.eval()
                if step >= m.n_steps:
                    break
                m.iteration += 1
                if done:  # game over, start again
                    obs = m.env.reset()
                    for skip in range(m.skip_start):  # skip the start of each game
                        obs, reward, done, info = m.env.step(0)
                    # state = util.preprocess_observation(obs)
                    state = obs
                # Actor evaluates what to do
                q_values = m.actor_q_values.eval(feed_dict={m.x: [state]})
                action = util.epsilon_greedy(q_values, step, m.n_outputs)
                # Actor plays
                obs, reward, done, info = m.env.step(action)
                # next_state = util.preprocess_observation(obs)
                next_state = obs
                # Let's memorize what just happened
                m.replay_memory.append((state, action, reward, next_state, 1.0 - done))
                state = next_state

                if m.iteration < m.training_start or m.iteration % m.training_interval != 0:
                    continue

                # Critic learns
                X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                    util.sample_memories(m.batch_size, m.replay_memory))
                next_q_values = m.actor_q_values.eval(
                    feed_dict={m.x: X_next_state_val})
                max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
                y_val = rewards + continues * m.discount_rate * max_next_q_values
                m.train_op.run(feed_dict={m.x: X_state_val, m.x_action: X_action_val, m.y: y_val})

                if current_high_score < np.max(y_val):
                    high_score = True
                    current_high_score = np.max(y_val)

                # Regularly copy critic to actor
                if step % m.copy_steps == 0 or high_score:
                    m.copy_critic_to_actor.run()

                # And save regularly
                if step % m.save_steps == 0 or high_score:
                    saver.save(sess, os.path.join(model_dir, checkpoint_path), global_step=m.global_step)

                print(
                    "Episode {} achieved a score of {} at {} training steps; current high score: {}".format(step, np.max(y_val), m.iteration, current_high_score))
                high_score = False
