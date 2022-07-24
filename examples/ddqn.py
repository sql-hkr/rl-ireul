import gym

from ireul.common.utils import mini_batch_train
from ireul.dqn.ddqn import DDQNAgent

env_id = "CartPole-v1"
MAX_STEPS = 100000
BATCH_SIZE = 32

env = gym.make(env_id)
agent = DDQNAgent(env, use_conv=False)
episode_rewards = mini_batch_train(env, agent, MAX_STEPS, BATCH_SIZE)

import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.arange(len(episode_rewards)), np.array(episode_rewards))
plt.show()
