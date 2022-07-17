import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import math
import gym


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size):
    episode_rewards = []
    train_done = False

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, next_state, reward, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            # <<< TODO: implement exactly
            if step % 10 == 0:
                agent.sync()
            # >>>

            if done or step == max_steps-1:

                # <<< TODO: implement exactly
                if episode_reward > 490:
                    train_done = True
                # >>>

                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

        # <<< TODO: implement exactly
        if train_done:
            break
        # >>>

    return episode_rewards