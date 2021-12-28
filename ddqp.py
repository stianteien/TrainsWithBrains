# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 12:21:35 2021

@author: Stian
"""

from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout, Input
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions),
                                      dtype=np.object)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]
        
        return states, actions, rewards, new_states, terminal
    
def build_dqp(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    
    inputs = Input(shape=input_dims)
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    
    output1 = Dense(n_actions, activation='softmax')(x)
    output2 = Dense(n_actions, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=[output1, output2])
    '''
    model = Sequential([
        
        Input(shape=input_dims),
        #LSTM(units=128, return_sequences=True),
        #Dropout(0.2),
        #LSTM(units=128),
        #Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(n_actions, activation='sigmoid')
        
        #Dense(fc1_dims, input_shape=(input_dims, ), activation='relu'),
        #Dense(fc2_dims, activation='relu'),
        #Dense(n_actions)              
        ])
    
    '''
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy')
    return model

class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, max_speed, epsilon, batch_size,
                 input_dims, epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=1000000, fname='ddqn_model.h5', replace_target=100):
        self.n_actions = n_actions
        self.max_speed = max_speed
        self.actions_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_name = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, False)
        self.q_eval = build_dqp(alpha, n_actions, input_dims, 256, 256)
        self.q_target = build_dqp(alpha, n_actions, input_dims, 256, 256)
        
    def remeber(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def choose_action(self, state):
        # Train: Return actions
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            #action = np.random.rand(1, self.n_actions) * self.max_speed
            #action = np.random.choice(self.actions_space)
            #action = np.random.rand(1, self.n_actions)[0]*self.speed
            action = np.array([np.random.choice([0,1], self.n_actions)])
            # Fiks actions her
        else:
            action = self.q_eval.predict(state)
            action = np.concatenate(action).argmax(axis=1)[np.newaxis,:]
            #action = np.argmax(actions)
        
        return action
    
    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                    self.memory.sample_buffer(self.batch_size)
            #action_values = np.array(self.actions_space, dtype=np.int8)
            #action_indices = np.dot(action, action_values)
            action_indexes = action.argmax(axis=0)
            action_indexes = action[:,1]
            
            
            self.q_next = self.q_target.predict(new_state) #pred nye speed
            self.q_eval1 = self.q_eval.predict(new_state)   #pred nye speed
            
            self.state = state
            self.action = action
            self.q_pred = self.q_eval.predict(state)       #pred speed
            #max_actions = np.argmax(q_eval, axis=1)
            
            self.q_target1 = self.q_pred.copy()
            
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            
            # For loop alle togene (med hver sin handling)
            for i in range(len(self.q_target1)):
                self.max_actions = np.argmax(self.q_eval1[i], axis=1)
            
                # Første output i  modellen
                self.q_target1[i][batch_index, list(action[:,i])] = reward + \
                    self.gamma*self.q_next[i][batch_index, self.max_actions.astype(int)]*done
            
            
            # Her læres det. Hva skal jeg anta er riktig?
            # Reward er jo en pekepin
            # q_target1 er det som skal være "riktig" - retning.
            # Høy reward forandre lite
            # Lav reward forandre mye - huber
            
            # Forsterke handling basert på reward!
            
            #self.q_target1[batch_index, :] = reward.reshape(reward.shape[0],1) + \
            #    self.gamma*self.q_next[batch_index, :]
            #self.q_target1[done == 0] = np.array([[0]*self.n_actions])
            
            #self.mod_reward = np.exp(-reward*0.01 + 4)
            #self.mod_reward = self.mod_reward.reshape(self.mod_reward.shape[0],1)
            
            
            #self.q_target1[batch_index, :] = self.mod_reward + \
            #    self.gamma*q_next[batch_index, :]    # done fjernet
            #self.q_target1[done == 0] = np.array([[0,0]])
            
            _ = self.q_eval.fit(state, self.q_target1, verbose=0)
            
            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                self.epsilon_min else self.epsilon_min
                
            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()
                
    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())
        
    def save_model(self, fname):
        self.q_eval.save(fname)
            
    def load_model(self, fname):
        self.q_eval = load_model(fname)
            
        if self.epsilon <= self.epsilon_min:
            self.update_network_parameters()
            
        
        
        
    
    
    
        