import gym

from ireul.common.utils import mini_batch_train
from ireul.dqn.noisy_dqn import NoisyDQNAgent

env_id = "CartPole-v1"
MAX_STEPS = 100000
BATCH_SIZE = 32
LR = 1e-3

env = gym.make(env_id)
agent = NoisyDQNAgent(env, use_conv=False, learning_rate=LR)
episode_rewards = mini_batch_train(env, agent, MAX_STEPS, BATCH_SIZE)

import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.arange(len(episode_rewards)), np.array(episode_rewards))
plt.show()
