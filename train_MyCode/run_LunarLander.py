# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:57:35 2019

@author: magicKD
"""

import gym
from gym import wrappers
from RL_model import Agent
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

env = gym.make('LunarLander-v2')
#env = env.unwrapped
env.seed(1)

#in order to close
import signal
import sys

np.random.seed(1)
tf.set_random_seed(1)

def CtrlCHandler(signum, frame):
    env.close()
    sys.exit(0)

N_A = env.action_space.n
N_S = env.observation_space.shape[0]
MEMORY_CAPACITY = 25000
TARGET_REP_ITER = 500
MAX_EPISODES = 10000
EPSILON_DECAY = 0.99#0.99941
EPSILON = 0.85;
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 32
HIDDEN = [400, 400]
RENDER = True

MINIMUM_REWARD = -250
agent = Agent(
    n_actions=N_A, n_features=N_S, learning_rate=LR, reward_decay=GAMMA,
    hidden=HIDDEN, batch_size=BATCH_SIZE, replace_target_iter=TARGET_REP_ITER,
    memory_size=MEMORY_CAPACITY, epsilon_decay=EPSILON_DECAY,training=False, loading=True, output_graph=True)


total_steps = 0
running_r = 0
r_scale = 100
average_reward = deque(maxlen = 20);

for i_episode in range(MAX_EPISODES):
    s = env.reset()  # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground, r_leg_on_ground)
    current_reward = 0
    
    for t in range(1000):
        signal.signal(signal.SIGINT, CtrlCHandler)
        env.render();
        a = agent.choose_action(s);
        s_, r, is_terminal, info = env.step(a)
        current_reward += r;
        agent.store_transition(s, a, r, s_, is_terminal);
        if (total_steps > MEMORY_CAPACITY):
            agent.learn()
        if is_terminal or current_reward < MINIMUM_REWARD:
            break;
        s = s_;
        total_steps += 1;
        
#        if total_steps % TARGET_REP_ITER == 0:
#            agent.update();
#    agent.update();
    
    average_reward.append(current_reward);
    print("%i, %.2f, %.2f, %.2f" % (i_episode, current_reward, np.average(average_reward), agent.epsilon))
