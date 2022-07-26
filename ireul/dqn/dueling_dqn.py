import math, random, copy

import numpy as np

import torch
from torch.nn import functional as F

from tensorboardX import SummaryWriter

from ireul.common.replay_buffers import ReplayMemory
from ireul.common.base_class import BaseAgent
from ireul.dqn.models import DuelingDQN


class DuelingDQNAgent(BaseAgent):
    def __init__(
            self,
            env,
            use_conv=True,
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.01,
            buffer_size=10000,
            eps_start=0.9,
            eps_end=0.05,
            eps_decay=200
        ):

        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayMemory(capacity=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_conv = use_conv
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.steps_done = 0

        if self.use_conv:
            pass
            # self.model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
            # self.target_model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        else:
            self.policy_net = DuelingDQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
            self.target_net = copy.deepcopy(self.policy_net)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters())

        
    def get_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(torch.tensor(np.expand_dims(state, 0)).to(self.device)).max(1)[1].item()
        else:
            return self.env.action_space.sample()

    def compute_loss(self, batch):
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device).unsqueeze(-1)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device).unsqueeze(-1)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device).unsqueeze(-1)

        curr_Q = self.policy_net.forward(state_batch).gather(1, action_batch)
        next_Q = self.target_net.forward(next_state_batch)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.unsqueeze(-1)
        expected_Q = reward_batch + (1 - done_batch) * self.gamma * max_next_Q
        
        loss = F.mse_loss(curr_Q, expected_Q.detach())
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def target_net_sync(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    