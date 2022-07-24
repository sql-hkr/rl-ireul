import gym
from .base_class import BaseAgent

from datetime import date, datetime
from tensorboardX import SummaryWriter

# TODO: implement a logger
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