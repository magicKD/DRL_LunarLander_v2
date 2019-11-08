# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:02:40 2019

@author: magicKD
"""

import numpy as np
import tensorflow as tf
from prioritized_replay import Memory
import pickle

np.random.seed(1)
tf.reset_default_graph()
tf.set_random_seed(1)

WEIGHTS_FILENAME = './weights/weights.h5'
memory_file_name = "./variable_value/replay";
FRAME_NUMBER = 4;
ACTION_NUMBER = 0

class Agent:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.99,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            epsilon_decay=None,
            hidden=[100, 50],
            output_graph=False,
            sess=None,
            training=True, 
            loading = False, 
            image_height = 85,
            image_width = 100,
            memory_refectch = False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features * FRAME_NUMBER
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.hidden = hidden
        self.epsilon_decay = epsilon_decay
        self.training = training;
        self.epsilon = 1 if self.training else 0.0
        
        self.image_height = image_height;
        self.image_width = image_width;

        self.learn_step_counter = 1
        self._build_net()
        
        if not memory_refectch:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = pickle.load(open(memory_file_name, "rb"))

        self.saver = tf.train.Saver();
        self.loading = loading;

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
            
        if loading:
            self._loadWeight();
            
        if not training:
            self._loadWeight();

        self.cost_his = []
        
    #保存训练好的神经网络
    def _saveWeight(self):
        if self.training:
            self.saver.save(self.sess, WEIGHTS_FILENAME);

    def _loadWeight(self):
        try:
            self.saver.restore(self.sess, WEIGHTS_FILENAME)
            print(self.sess)
        except Exception as e:
            print("Error loading agent weights from disk.", e);
            
            
    '''
    CNN function
    '''
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def _build_net(self):
        def build_layers(s, c_names, w_initializer, b_initializer):
            regularizer = tf.contrib.layers.l1_regularizer(0.05);
            
            s = tf.reshape(s, [-1, self.image_height, self.image_width, FRAME_NUMBER])
            
            W_conv1 = self.weight_variable([11, 11, 4, 32]);
            b_conv1 = self.bias_variable([32]);
            
            h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 6) + b_conv1);
            
            h_pool1 = self.max_pool_2x2(h_conv1);
            
            W_conv2 = self.weight_variable([7, 7, 32, 64]);
            b_conv2 = self.bias_variable([64]);
            
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2);
            
            W_conv3 = self.weight_variable([5, 5, 64, 64]);
            b_conv3 = self.bias_variable([64]);
            h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3);
            
            h_conv3_shape = h_conv3.get_shape().as_list();
            print(h_conv3_shape);#[None, 6, 11, 64]
            flatten_size = h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3];
            
            h_conv3_flat = tf.reshape(h_conv3, [-1, flatten_size]);
            
            
            for i, h in enumerate(self.hidden):
                if i == 0:
                    in_units, out_units, inputs = flatten_size, self.hidden[i], h_conv3_flat
                else:
                    in_units, out_units, inputs = self.hidden[i-1], self.hidden[i], l
                with tf.variable_scope('l%i' % i):
                    w = tf.get_variable('w', [in_units, out_units], initializer=w_initializer, collections=c_names, regularizer=regularizer)
                    b = tf.get_variable('b', [1, out_units], initializer=b_initializer, collections=c_names, regularizer=regularizer)
                    c = tf.get_variable('c', [ACTION_NUMBER, out_units], initializer=b_initializer, collections=c_names, regularizer=regularizer)
                    l = tf.nn.relu(tf.matmul(inputs, w) + b)
                    
            with tf.variable_scope('Value'):
                w = tf.get_variable('w', [self.hidden[-1], 1], initializer=w_initializer, collections=c_names)
                b = tf.get_variable('b', [1, 1], initializer=b_initializer, collections=c_names)
                self.V = tf.matmul(l, w) + b

            with tf.variable_scope('Advantage'):
                w = tf.get_variable('w', [self.hidden[-1], self.n_actions], initializer=w_initializer, collections=c_names)
                b = tf.get_variable('b', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.A = tf.matmul(l, w) + b

            with tf.variable_scope('Q'):
                out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)

            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
#        self.a = tf.placeholder(tf.float32, [None, ACTION_NUMBER], name='s');
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.01), tf.constant_initializer(0.01)  # config of layers

            self.q_eval = build_layers(self.s, c_names, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.abs_errors = tf.abs(tf.reduce_sum(self.q_target - self.q_eval, axis=1))  # for updating Sumtree
            self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
#            self._train_op = tf.train.RMSPropOptimizer(self.lr,0.99,0.0,1e-6).minimize(self.loss);
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_, is_terminal ):
        s_ = s_.reshape(1, self.image_height * self.image_width, 1)
        newState = np.append(s_, self.currentState[:,:,1:], axis=2)
#        action = self.currentAction.tolist();
#        action.insert(0, a)
#        action.pop();
#        self.currentAction = np.array(action)
        transition = np.hstack((self.currentState.flatten(), [a, r], newState.flatten(), is_terminal));
        self.currentState = newState;
#        transition = np.hstack((s, [a, r], s_, is_terminal))
        max_p = np.max(self.memory.tree.tree[-self.memory.tree.capacity:])
        self.memory.store(max_p, transition)

    def choose_action(self):
#        observation = observation[np.newaxis, :]
#        newState = np.append(observation, self.currentState[:,:,1:])
        if np.random.uniform() > self.epsilon or (not self.training):
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: self.currentState.reshape((1, -1))})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        
    def _update(self):
        self._replace_target_params();
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay;
            self.lr *= self.epsilon_decay;
#        self._saveWeight();    

    def setInitState(self, observation):
        observation = observation[np.newaxis, :]
        self.currentState = np.stack((observation, observation, observation, observation), axis = 2);
        self.currentAction = np.zeros((ACTION_NUMBER,))

    def learn(self):
#        if self.learn_step_counter % self.replace_target_iter == 0:
#            self._replace_target_params();
##            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
#            self.epsilon = 1 - (1 - self.epsilon) * 0.98 if self.epsilon < self.epsilon_max else self.epsilon_max
#            self._saveWeight();
        if not self.training:
            return ;
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._update();
        if self.learn_step_counter % (self.replace_target_iter * 10) == 0:
            self._saveWeight();  
#            pickle.dump(self.memory, open(memory_file_name, "wb"));
        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)




        # double DQN
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features-1:-1],  # next observation
                       self.s: batch_memory[:, -self.n_features-1:-1],
                       })  # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        max_act4next = np.argmax(q_eval4next,
                                 axis=1)  # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions


        for i in range(self.batch_size):
            terminal = batch_memory[batch_index[i]][-1];
            if terminal:
                q_target[batch_index[i], eval_act_index[i]] = reward[i];
            else:
                q_target[batch_index[i], eval_act_index[i]] = reward[i] + self.gamma * selected_q_next[i]

#        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next


        #优化一次        
        _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                 feed_dict={self.s: batch_memory[:, :self.n_features],
                                                            self.q_target: q_target,
                                                            self.ISWeights: ISWeights,})
        for i in range(len(tree_idx)):  # update priority
            idx = tree_idx[i]
            self.memory.update(idx, abs_errors[i])

        self.cost_his.append(self.cost)

#        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


