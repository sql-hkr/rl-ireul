import math, random, copy, datetime
from collections import namedtuple, deque

import gym

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from tensorboardX import SummaryWriter


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
    def forward(self, x):
        return self.layers(x)

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 100
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# summary writer
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter()

env = gym.make("CartPole-v1")
n_inputs = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(n_inputs, n_actions).to(device)
target_net = copy.deepcopy(policy_net)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(1000)
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(torch.tensor(np.expand_dims(state, 0)).to(device)).max(1)[1].item()
    else:
        return env.action_space.sample()



all_rewards = []
total_reward = 0
episode = 1

state = env.reset()

for step in range(30000):
    action = select_action(state)
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    memory.push(state, action, next_state, reward, done)
    state = next_state

    if len(memory) >= BATCH_SIZE:
        batch = memory.sample(BATCH_SIZE)
        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(batch.action).to(device).unsqueeze(-1)
        reward_batch = torch.FloatTensor(batch.reward).to(device).unsqueeze(-1)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done_batch = torch.FloatTensor(batch.done).to(device).unsqueeze(-1)

        curr_Q = policy_net.forward(state_batch).gather(1, action_batch)
        next_Q = target_net.forward(next_state_batch)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = reward_batch + (1 - done_batch) * GAMMA * max_next_Q
        
        loss = F.mse_loss(curr_Q, expected_Q.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    if done:
        state = env.reset()
        print(f"step: {step:5d} | reward: {total_reward:3.0f}")
        writer.add_scalar("reward", total_reward, episode)

        all_rewards.append(total_reward)

        # end when the number of steps reaches 490 or more
        if total_reward > 490:
            break
        
        total_reward = 0
        episode += 1

    if step % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

plt.plot(np.arange(len(all_rewards)), np.array(all_rewards))
plt.show()

writer.close()
