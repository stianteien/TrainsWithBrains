# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:08:32 2021

@author: Stian
"""

from train import Train
from railway import Railway
from subwaysystem import SubwaySystem
import pandas as pd
import numpy as np
import random
from ddqp import DDQNAgent
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt


linje10coords = pd.read_csv("lines/linje10.csv", index_col=0)
linje11coords = pd.read_csv("lines/linje11.csv", index_col=0)
linje12coords = pd.read_csv("lines/linje12.csv", index_col=0)

#linje10stops = pd.concat([linje10coords[linje10coords.index==80]])
#linje10stops[["stop_time", "active"]] = [[100,False]]

#linje11stops = pd.concat([linje11coords[linje11coords.index==90]])
#linje11stops[["stop_time", "active"]] = [[150,False]]


subwaysystem = SubwaySystem(h=300, w=200)

linje10 = Railway(linje10coords, train_loop_strategy="line")
linje11 = Railway(linje11coords, train_loop_strategy="line")
linje12 = Railway(linje12coords, train_loop_strategy="line")


linje10.add_train(Train())
linje11.add_train(Train())
linje12.add_train(Train())

subwaysystem.add_railway(linje10)
subwaysystem.add_railway(linje11)
#subwaysystem.add_railway(linje12)


#subwaysystem.find_intersections()
#subwaysystem.render()
#subwaysystem.run_simualation()
#subwaysystem.reset()



# ========
# Reinformant learning!
# ========

agent = DDQNAgent(alpha=0.005, gamma=0.99, n_actions=2, max_speed=100,
                  epsilon=1.0, batch_size=32, input_dims=8, epsilon_end=0.3)
agent1 = DDQNAgent(alpha=0.005, gamma=0.99, n_actions=2, max_speed=100,
                  epsilon=1.0, batch_size=32, input_dims=8, epsilon_end=0.3)

linje10.trains[0].agent = agent
linje11.trains[0].agent = agent1

r_history = []

n_games = 10
n_interact = 200
done = False
max_interations = 10000
reward_h = []


for i in range(n_games):
    score = 0
    o = 0
    state, done = subwaysystem.reset()
    
    actions = []
    states = []
    states_ = []
    rewards = []
    dones = []
    speeds = []
    distances = []
        
        # Do action on env
    while not done:
        if o < n_interact:
            action = subwaysystem.logic_movement()
        else:
            action0 = linje10.trains[0].agent.choose_action(state)
            action1 = linje11.trains[0].agent.choose_action(state)
            
            action = [[action0, action1]]
            
        actions.append(action)
            
        


        n = 25
        for _ in range(n):
            o += 1
            state_, reward, done, info = subwaysystem.step(action)
            score += reward
            
            # Save things on the way
            rewards.append([linje10.trains[0].reward,
                            linje11.trains[0].reward])
            speeds.append([train.speed for train,_,_ in subwaysystem.trains])
            distances.append(pdist([train.position for train,_,_ in subwaysystem.trains]))
            if o>n_interact:
                linje10.trains[0].agent.remeber(state, action0, 
                                                linje10.trains[0].reward,
                                                state_, done)
                linje11.trains[0].agent.remeber(state, action0, 
                                                linje11.trains[0].reward,
                                                state_, done)
                
            
            #subwaysystem.save_image(o)
            state = state_

        if o>n_interact+1:
            linje10.trains[0].agent.learn()
            linje11.trains[0].agent.learn()
        
        #if o>100:
        #    break
        #if o%300 == 0:
        #    print(o, [train.speed for train,_,_ in subwaysystem.trains], np.round(reward,3))
        
        if done:
            reward_h.append(np.sum(np.mean(np.array(rewards), axis=1)))
            print(f"Collisions. Reward: {round(np.sum(np.mean(np.array(rewards), axis=1)),3)}. Last 10 mean: {round(np.mean(reward_h[-10:]),3)}")
            
        if o>max_interations:
            done = True
            reward_h.append(np.sum(np.mean(np.array(rewards), axis=1)))
            print(f"stopper pga maxinteretatoion, reward {round(np.sum(np.mean(np.array(rewards), axis=1)),3)}. Last 10 mean: {round(np.mean(reward_h[-10:]),3)}")
            
    
            
        

        # == Learn from action ==
        #agent.discount_reward()
        # agent.learn(tape)
        
        #r_history.append(score)
        

distances = np.array(distances)
speeds = np.array(speeds)
actions = np.array(actions).reshape(len(actions),2)
rewards = np.array(rewards)


