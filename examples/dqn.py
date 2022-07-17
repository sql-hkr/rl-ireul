import gym

from ireul.common.utils import mini_batch_train
from ireul.dqn.dqn import DQNAgent

env_id = "CartPole-v1"
MAX_EPISODES = 1000
MAX_STEPS = 500
BATCH_SIZE = 32

env = gym.make(env_id)
agent = DQNAgent(env, use_conv=False)
episode_rewards = mini_batch_train(env, agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE)

import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.arange(len(episode_rewards)), np.array(episode_rewards))
plt.show()
