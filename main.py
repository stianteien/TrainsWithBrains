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



linje7coords = pd.read_csv("lines/linje7.csv", index_col=0)
linje8coords = pd.read_csv("lines/linje8.csv", index_col=0)
#linje12coords = pd.read_csv("lines/linje12.csv", index_col=0)

#linje10stops = pd.concat([linje10coords[linje10coords.index==80]])
#linje10stops[["stop_time", "active"]] = [[100,False]]

#linje11stops = pd.concat([linje11coords[linje11coords.index==90]])
#linje11stops[["stop_time", "active"]] = [[150,False]]


subwaysystem = SubwaySystem(h=300, w=200)

linje7 = Railway(linje7coords, train_loop_strategy="line")
linje8 = Railway(linje8coords, train_loop_strategy="line")


linje7.add_train(Train())
linje8.add_train(Train())


subwaysystem.add_railway(linje7)
subwaysystem.add_railway(linje8)
#subwaysystem.add_railway(linje12)


#subwaysystem.find_intersections()
#subwaysystem.render()
#subwaysystem.run_simualation()
#subwaysystem.reset()



# ========
# Reinformant learning!
# ========

agent = DDQNAgent(alpha=0.005, gamma=0.99, n_actions=2, max_speed=100,
                  epsilon=1.0, batch_size=32, input_dims=4, epsilon_end=0.3)
agent1 = DDQNAgent(alpha=0.005, gamma=0.99, n_actions=2, max_speed=100,
                  epsilon=1.0, batch_size=32, input_dims=8, epsilon_end=0.3)

linje7.trains[0].agent = agent
linje8.trains[0].agent = agent1

r_history = []

n_games = 100
n_interact = 200
done = False
max_interations = 5000
reward_h = []
speeds_h = []


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
            action0 = linje7.trains[0].agent.choose_action(linje7.trains[0].state)
            action1 = 1#linje8.trains[0].agent.choose_action(state)
            
            action = [[action0, action1]]
            
        actions.append(action)
            
        


        n = 25
        j = 0
        while j<=n:
        #for _ in range(n): # When done totally break out!
            o += 1
            save_state = linje7.trains[0].state
            state_, reward, done, info = subwaysystem.step(action)
            score += reward
            
            # Save things on the way
            rewards.append([linje7.trains[0].reward,
                            linje8.trains[0].reward])
            speeds.append([train.speed for train,_,_ in subwaysystem.trains])
            distances.append(pdist([train.position for train,_,_ in subwaysystem.trains]))
            if o>n_interact:
                linje7.trains[0].agent.remeber(save_state, action[0][0], 
                                                linje7.trains[0].reward,
                                                linje7.trains[0].state, # <- new state_ (new state)
                                                done)
                #linje8.trains[0].agent.remeber(state, action0, 
                #                                linje8.trains[0].reward,
                #                                state_, done)
                
            
            #subwaysystem.save_image(o)
            state = state_
            
            # Update j
            j += 1
            
            #Break out if done
            if subwaysystem.done:
                done = True
                j = n + 1


        if o>n_interact+1:
            linje7.trains[0].agent.learn()
            #linje8.trains[0].agent.learn()
        
        #if o>100:
        #    break
        #if o%300 == 0:
        #    print(o, [train.speed for train,_,_ in subwaysystem.trains], np.round(reward,3))
        
        if done:
            reward_h.append(np.sum(np.mean(np.array(rewards), axis=1)))
            #print(f"Collisions. Reward: {round(np.sum(np.mean(np.array(rewards), axis=1)),3)}. Last 10 mean: {round(np.mean(reward_h[-10:]),3)}")
            print(f"Done. Amount of steps: {subwaysystem.counter}")
            
        if o>max_interations:
            done = True
            reward_h.append(np.sum(np.mean(np.array(rewards), axis=1)))
            print(f"stopper pga maxinteretatoion, reward {round(np.sum(np.mean(np.array(rewards), axis=1)),3)}. Last 10 mean: {round(np.mean(reward_h[-10:]),3)}")
            print(f"Amount of steps: {subwaysystem.counter}")
    
            
    speeds_h.append(speeds)
    plt.plot(np.array(speeds))
    plt.title("speeds")
    plt.show()
    plt.plot(np.array(rewards))
    plt.title("rewards")
    plt.show()

        # == Learn from action ==
        #agent.discount_reward()
        # agent.learn(tape)
        
        #r_history.append(score)
        

distances = np.array(distances)
speeds = np.array(speeds)
actions = np.array(actions).reshape(len(actions),2)
rewards = np.array(rewards)
speeds_h = np.array(speeds_h, dtype=object)



#plt.plot(m[:,0])
#plt.fill_between(m[:,0], m[:,0]+s[:,0], m[:,0]-s[:,0], alpha=0.5)



