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

import cv2
import base64
import imageio

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
    
    
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 200
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

def imagePreprocess(pixels):
#    pixels = cv2.medianBlur(pixels, 9)
#    s_pixels_ = cv2.cvtColor(cv2.resize(pixels, (150, 100)), cv2.COLOR_BGR2GRAY)[:85, 25:125]
#    ret, s_pixels_ = cv2.threshold(s_pixels_, 1, 255, cv2.THRESH_BINARY);
#    s_pixels_ = np.reshape(s_pixels_, (IMAGE_HEIGHT, IMAGE_WIDTH, 1));
    s_pixels_ = cv2.resize(pixels, (300, 200))
    s_pixels_ = cv2.cvtColor(s_pixels_, cv2.COLOR_BGR2GRAY)
    s_pixels_ = np.reshape(s_pixels_, (IMAGE_HEIGHT, IMAGE_WIDTH, 1));
    s_pixels_ = s_pixels_.flatten();
    return s_pixels_

N_A = env.action_space.n
N_S = env.observation_space.shape[0]
MEMORY_CAPACITY = 1000
TARGET_REP_ITER = 500
MAX_EPISODES = 10000
EPSILON_DECAY = 0.99#0.99941
EPSILON = 0.85;
GAMMA = 0.99
LR = 0.0005
BATCH_SIZE = 32
HIDDEN = [400, 400]
RENDER = True

MINIMUM_REWARD = -250

MEMORY_REFECTCH = False


agent = Agent(
    n_actions=N_A, n_features=IMAGE_SIZE, learning_rate=LR, reward_decay=GAMMA,
    hidden=HIDDEN, batch_size=BATCH_SIZE, replace_target_iter=TARGET_REP_ITER,
    memory_size=MEMORY_CAPACITY, epsilon_decay=EPSILON_DECAY,training=False, loading=True, output_graph=False, 
    memory_refectch=MEMORY_REFECTCH,image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH)


total_steps = 0
if MEMORY_REFECTCH:
    total_steps = MEMORY_CAPACITY;
    
running_r = 0
r_scale = 100
average_reward = deque(maxlen = 20);

seeds = [43,44,45,46,47]

agent.epsilon = 0.0
for i_episode in range(MAX_EPISODES):
    env.seed(seeds[i_episode % len(seeds)]);
    s = env.reset()  # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground, r_leg_on_ground)
    current_reward = 0
    pixels = env.render(mode="rgb_array");
    s_pixels = imagePreprocess(pixels);
    agent.setInitState(s_pixels)
#    agent.lr = LR * 
    for t in range(40):
        a = 0;
        s_, r, is_terminal, info = env.step(a)
        s_pixels = env.render(mode="rgb_array");
        s_pixels = imagePreprocess(s_pixels);
    for t in range(1000):
        signal.signal(signal.SIGINT, CtrlCHandler)
#        env.render();
        a = agent.choose_action();
        s_, r, is_terminal, info = env.step(a)
        s_next_pixels = env.render(mode="rgb_array");
        s_next_pixels = imagePreprocess(s_next_pixels);
#        if r < 0 and (a == 0 or a == 2):
#            r *= 1.1
#        if is_terminal and (s[6] != 0 or s[7] != 0):
#            r += 30
        current_reward += r;
        #debug propose
        agent.store_transition(s_pixels, a, r, s_next_pixels, is_terminal);
#        if (total_steps > MEMORY_CAPACITY):
#            agent.learn()
        if is_terminal or current_reward < MINIMUM_REWARD:
            break;
        s = s_;
        total_steps += 1;
        
        s_pixels = s_next_pixels
        
#        if total_steps % TARGET_REP_ITER == 0:
#            agent.update();
#    agent.update();
    
    average_reward.append(current_reward);
    print("%i, %.2f, %.2f, %.2f" % (i_episode, current_reward, np.average(average_reward), agent.epsilon))

for i in range(500):
    agent.learn()
    
res = []
for i in range(len(agent.cost_his) // 1000 - 1):
    res.append(np.average(agent.cost_his[i*1000:(i+1)*1000]))

