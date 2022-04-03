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


linje10coords = pd.read_csv("lines/linje10.csv", index_col=0)
linje11coords = pd.read_csv("lines/linje11.csv", index_col=0)
linje12coords = pd.read_csv("lines/linje12.csv", index_col=0)

linje10stops = pd.concat([linje10coords[linje10coords.index==80]])
linje10stops[["stop_time", "active"]] = [[100,False]]

linje11stops = pd.concat([linje11coords[linje11coords.index==90]])
linje11stops[["stop_time", "active"]] = [[150,False]]


subwaysystem = SubwaySystem(h=300, w=200)

linje10 = Railway(linje10coords, linje10stops, train_loop_strategy="line")
linje11 = Railway(linje11coords, linje11stops, train_loop_strategy="line")
linje12 = Railway(linje12coords, train_loop_strategy="line")


linje10.add_train(Train())
linje11.add_train(Train())
linje12.add_train(Train())

subwaysystem.add_railway(linje10)
subwaysystem.add_railway(linje11)
subwaysystem.add_railway(linje12)


#subwaysystem.find_intersections()
#subwaysystem.render()
#subwaysystem.run_simualation()
#subwaysystem.reset()


# ========
# Reinformant learning!
# ========

agent = DDQNAgent(alpha=0.005, gamma=0.99, n_actions=2, max_speed=100,
                  epsilon=1.0, batch_size=32, input_dims=8, epsilon_end=0.3)
r_history = []

n_games = 50
done = False
max_interations = 10000


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
        action = agent.choose_action(state)


        n = 50
        for _ in range(n):
            o += 1
            state_, reward, done, info = subwaysystem.step(action)
            score += reward
            
            # Save things on the way
            rewards.append(reward)
            speeds.append([train.speed for train,_,_ in subwaysystem.trains])
            distances.append(pdist([train.position for train,_,_ in subwaysystem.trains]))
            agent.remeber(state, action, reward, state_, done)
            
            subwaysystem.save_image(o)
            state = state_

        
        agent.learn()
        
        #if o>100:
        #    break
        if o%300 == 0:
            print(o, [train.speed for train,_,_ in subwaysystem.trains], np.round(reward,3))
        
        if done:
            print(f"krasj, , reward {sum(rewards)} ")
            
        if o>max_interations:
            done = True
            print(f"stopper pga maxinteretatoion, reward {sum(rewards)}")
            
    
            
        

        # == Learn from action ==
        #agent.discount_reward()
        # agent.learn(tape)
        
        #r_history.append(score)
        

distances = np.array(distances)
speeds = np.array(speeds)


