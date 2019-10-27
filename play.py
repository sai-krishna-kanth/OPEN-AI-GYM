import numpy as np
from agent import DDQN
import matplotlib.pyplot as plt
import gym

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
import datetime as dt

def play(env_name, agent, episodes = 1000, train=True, until=0, render=False):
    if train:
        train_writer = tf.summary.create_file_writer("tensorboard/" + env +
                                                     f"/_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
                                                    )
    if render:
        env = wrappers.Monitor(gym.make(env_name), './movies/', force=True)
        env.file_prefix = env_name
        env.file_infix = ''
    else:
        env = gym.make(env)
    returns = []
    for i in range(episodes):
        state = env.reset()
        g = 0
        avg_loss = 0
        n = 0
        while True:
            n += 1
            if render:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            if train:
                loss = agent.train(state, action, reward, next_state, done)
                avg_loss += (1/n)*(loss-avg_loss)
            g += reward
            if done:
                break
            state = next_state
        returns.append(g)
        if i > 100:
            # most recent 100 consecutive episodes
            avg_reward = np.mean(returns[-100:])
        else:
            avg_reward = np.mean(returns)

        if not ((i+1) % 100):
            print(i+1, avg_reward)

        if train:
            with train_writer.as_default():
                tf.summary.scalar('avg_reward', avg_reward, step=i)
                tf.summary.scalar('avg loss', avg_loss, step=i)
        if until and avg_reward > until:
          break
    return (avg_reward, i) if train else avg_reward

def train(env):
    if env == "Cartpole-v0":
        agent = DDQN(4,2)
        play(env,agent,until=195)
    else:
        agent = DDQN(state_dim=(210,160,3), n_actions=9)
        play(env,agent)
    agent.save_model(env)

def display(env):
    if env == "CartPole-v0":
        agent = DDQN(4, 2, eps_max=0, load_path="models/"+env+"_model.h5")
        play(env,agent,train=False,render=True,episodes=1)
    else:
        agent = DDQN(state_dim=(210,160,3), n_actions=9, eps_max=0,
                     load_path="models/"+env+"_model.h5")
        play(env,agent,train=False,render=True,episodes=1)
                                                
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DDQN on Cartpole or MsPacman')
    parser.add_argument('--pac', action="store_true",
                        help='Play Pacman; Default is Cartpole')
    parser.add_argument('-t', action='store_true',
                        help='train DDQN; default is display trained DDQN')

    args = parser.parse_args()
    env = "MsPacman-v0" if args.pac else "CartPole-v0"
    if args.t:
        train(env)
    else:
        display(env)