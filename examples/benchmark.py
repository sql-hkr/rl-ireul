import gym

from ireul.common.utils import mini_batch_train
from ireul.dqn.dqn import DQNAgent
from ireul.dqn.ddqn import DDQNAgent

from collections import namedtuple
import copy

env_id = "CartPole-v1"
MAX_STEPS = 100000
BATCH_SIZE = 32

history = []
for i in range(3):
    env = gym.make(env_id)
    agent = DDQNAgent(env, use_conv=False)
    episode_rewards1 = mini_batch_train(env, agent, MAX_STEPS, BATCH_SIZE)

    env = gym.make(env_id)
    agent = DDQNAgent(env, use_conv=False)
    episode_rewards2 = mini_batch_train(env, agent, MAX_STEPS, BATCH_SIZE)

    history.append((
        copy.deepcopy(episode_rewards1),
        copy.deepcopy(episode_rewards2)
    ))
    

import numpy as np
import matplotlib.pyplot as plt


for indx, h in enumerate(history):
    dqn = h[0]
    ddqn = h[1]
    plt.plot(np.arange(len(dqn)), np.array(dqn), label=f"alg1: #{indx}")
    plt.plot(np.arange(len(ddqn)), np.array(ddqn), label=f"alg2: #{indx}")
plt.legend()
plt.xlabel("eps")
plt.ylabel("reward")
plt.show()
