import gym
from .base_class import BaseAgent

import math
from datetime import date, datetime

import torch
from torch import nn
from torch.nn import functional as F

from tensorboardX import SummaryWriter

writer = SummaryWriter()

def mini_batch_train(
    env: gym.Env,
    agent: BaseAgent,
    max_steps: int,
    batch_size: int
    ):

    episode_rewards = []

    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        env.render()
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, next_state, reward, done)
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)   

        if step % 10 == 0:
            agent.target_net_sync()

        if done or step == max_steps-1:
            episode_rewards.append(episode_reward)
            print("Episode " + str(len(episode_rewards)) + ": " + str(episode_reward))
            
            writer.add_scalar("reward", episode_reward, step)
            
            if episode_reward > 490:
                break

            # reset
            state = env.reset()
            episode_reward = 0
            continue

        state = next_state

    writer.close()
    
    return episode_rewards

class NoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(NoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.is_training = is_training

        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer("epsilon_weight", torch.FloatTensor(num_out, num_in)) 
        self.register_buffer("epsilon_bias", torch.FloatTensor(num_out))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()

        if self.is_training:
            weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
            bias = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        std = math.sqrt(3 / self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std,std)

        self.sigma_weight.data.fill_(0.017)
        self.sigma_bias.data.fill_(0.017)

    def reset_noise(self):
        self.epsilon_weight.data.normal_()
        self.epsilon_bias.data.normal_()

    
class FactorizedNoisyLinear(nn.Module):

    def __init__(self, num_in, num_out, is_training=True):
        super(FactorizedNoisyLinear, self).__init__()
        self.num_in = num_in
        self.num_out = num_out 
        self.is_training = is_training

        self.mu_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.mu_bias = nn.Parameter(torch.FloatTensor(num_out)) 
        self.sigma_weight = nn.Parameter(torch.FloatTensor(num_out, num_in))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(num_out))
        self.register_buffer("epsilon_i", torch.FloatTensor(num_in))
        self.register_buffer("epsilon_j", torch.FloatTensor(num_out))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        self.reset_noise()
        
        if self.is_training:
            epsilon_weight = self.epsilon_j.ger(self.epsilon_i)
            epsilon_bias = self.epsilon_j
            weight = self.mu_weight + self.sigma_weight * epsilon_weight
            bias = self.mu_bias + self.sigma_bias * epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        std = 1 / math.sqrt(self.num_in)
        self.mu_weight.data.uniform_(-std, std)
        self.mu_bias.data.uniform_(-std, std)

        self.sigma_weight.data.fill_(0.5 / math.sqrt(self.num_in))
        self.sigma_bias.data.fill_(0.5 / math.sqrt(self.num_in))

    def reset_noise(self):
        eps_i = torch.randn(self.num_in)
        eps_j = torch.randn(self.num_out)
        self.epsilon_i = eps_i.sign() * (eps_i.abs()).sqrt()
        self.epsilon_j = eps_j.sign() * (eps_j.abs()).sqrt()
