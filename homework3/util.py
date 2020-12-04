import numpy as np
import numpy.random as rnd
# Helper Functions

def sample_memories(batch_size, replay_memory):
    indices = rnd.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
            cols[4].reshape(-1, 1))


eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 50000


def epsilon_greedy(q_values, step, n_outputs):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if rnd.rand() < epsilon:
        return rnd.randint(n_outputs)  # random action
    else:
        return np.argmax(q_values)  # optimal action
