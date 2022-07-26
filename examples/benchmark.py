from collections import namedtuple
import copy
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

import gym

from ireul.common.utils import mini_batch_train
from ireul.common.base_class import BaseAgent
from ireul.dqn.dqn import DQNAgent
from ireul.dqn.ddqn import DDQNAgent
from ireul.dqn.noisy_dqn import NoisyDQNAgent
from ireul.dqn.dueling_dqn import DuelingDQNAgent

env_id = "CartPole-v1"
MAX_STEPS = 100000
BATCH_SIZE = 32
LR = 1e-3

agents: Dict[str, BaseAgent] = {
    "DQN": DQNAgent,
    "DDQN": DDQNAgent,
    "NoisyDQN": NoisyDQNAgent,
    "DuelingDQN": DuelingDQNAgent
    }

for name, agent in agents.items():
    env = gym.make(env_id)
    A = agent(env, use_conv=False, learning_rate=LR)
    episode_rewards = mini_batch_train(env, A, MAX_STEPS, BATCH_SIZE)
    plt.plot(np.arange(len(episode_rewards)), np.array(episode_rewards), label=name)

plt.legend()
plt.xlabel("eps")
plt.ylabel("reward")
plt.show()
