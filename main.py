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
from datetime import datetime


linje7coords = pd.read_csv("lines/linje7.csv", index_col=0)
linje8coords = pd.read_csv("lines/linje8.csv", index_col=0)
linje9coords = pd.read_csv("lines/linje9.csv", index_col=0)
#linje10coords = pd.read_csv("lines/linje11.csv", index_col=0)
#linje11coords = pd.read_csv("lines/linje11.csv", index_col=0)


#linje10stops = pd.concat([linje10coords[linje10coords.index==80]])
#linje10stops[["stop_time", "active"]] = [[100,False]]

#linje11stops = pd.concat([linje11coords[linje11coords.index==90]])
#linje11stops[["stop_time", "active"]] = [[150,False]]



linje7 = Railway(linje7coords, train_loop_strategy="line")
linje8 = Railway(linje8coords, train_loop_strategy="line")
linje9 = Railway(linje9coords, train_loop_strategy="line")
#linje10 = Railway(linje10coords, train_loop_strategy="line")
#linje11 = Railway(linje11coords, train_loop_strategy="line")


linje7.add_train(Train())
linje8.add_train(Train())
linje9.add_train(Train())
# linje10.add_train(Train())
# linje11.add_train(Train())



subwaysystem = SubwaySystem(h=300, w=200)

subwaysystem.add_railway(linje7)
subwaysystem.add_railway(linje8)
subwaysystem.add_railway(linje9)
# subwaysystem.add_railway(linje10)
# subwaysystem.add_railway(linje11)


#subwaysystem.find_intersections()
#subwaysystem.render()
#subwaysystem.run_simualation()
#subwaysystem.reset()

subwaysystem.reorder_trains()



# ========
# Reinformant learning!
# ========
counter = pd.read_csv("counter.csv", index_col = 0)

if not counter.history_fname.any():
    counter.history_fname = ["history-" + datetime.now().strftime('%Y%m%d-%H%M') + ".csv"]
    print("Setting new fname")


# Add brain to all trains
i = 0
for train,_,_ in subwaysystem.trains:
    train.agent = DDQNAgent(alpha=0.005, gamma=0.99, n_actions=2, max_speed=100,
                                epsilon=1.0, batch_size=32, input_dims=4, epsilon_end=0.03,
                                fname=f'agent{i}.h5')
        
    if counter.n[0] < counter.goal[0] and counter.n[0] != 0:
        print(f"loading model from data {train.agent.fname}")
        train.agent.load_model(train.agent.fname)            
            
    i+=1


r_history = []
history = []

n_games = counter.goal[0] + 1
n_interact = 200
done = False
max_interations = 3000
reward_h = []
speeds_h = []


for i in range(n_games - counter.n[0]):
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
            action = []
            for train,_,_ in subwaysystem.trains:
                action.append(train.agent.choose_action(train.state))
                train.action = action

            action = [action]
            
        actions.append(action)
            
        


        n = 25
        j = 0
        while j<=n:
        #for _ in range(n): # When done totally break out!
            o += 1
            save_states = []
            for train,_,_ in subwaysystem.trains:
                save_states.append(train.state)
                train.save_state = train.state
            
            
            state_, reward, done, info = subwaysystem.step(action)
            score += reward
            
            # Save things on the way
            rewards.append([train.reward for train,_,_ in subwaysystem.trains])
            speeds.append([train.speed for train,_,_ in subwaysystem.trains])
            distances.append(pdist([train.position for train,_,_ in subwaysystem.trains]))
            
            if o>n_interact:
                for train,_,_ in subwaysystem.trains:
                    if not train.reached_end:
                        train.agent.remeber(train.save_state, train.action, 
                                                    train.reward,
                                                    train.state, # <- new state_ (new state)
                                                    done)
                    
            
            #subwaysystem.save_image(o)
            state = state_
            
            # Update j
            j += 1
            
            #Break out if done
            if subwaysystem.done:
                done = True
                j = n + 1


        if o>n_interact+1:
            for train,_,_ in subwaysystem.trains:
                if not train.reached_end:
                    train.agent.learn()
        
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
            subwaysystem.done_flag = "max_iter"
    
            
    speeds_h.append(speeds)
    print(f"{counter.n[0]} of {n_games-1}")
    for train,_,_ in subwaysystem.trains:
        train.agent.save_model(train.agent.fname)  
    

        
    
    # Save amount of steps.
    history.append([subwaysystem.done_flag, counter.n[0], subwaysystem.counter])
    #history[subwaysystem.done_flag].append([i, subwaysystem.counter])
    #history["all"].append(subwaysystem.counter)
    
    
    #Save to file
    if counter.n[0] == 0:
        print(f"Making a new file")
        df = pd.DataFrame(data=history, columns=["type", "i", "count"])
        df.to_csv(counter.history_fname[0])
        
    else:
        
        df = pd.read_csv(counter.history_fname[0], index_col=0)
        df.loc[len(df)] = history[-1]
        df.to_csv(counter.history_fname[0])
    
    counter.n += 1
    counter.to_csv("counter.csv")
    
    if counter.n[0] > counter.goal[0]:
        counter.n = 0
        counter.history_fname = [None]
        counter.to_csv("counter.csv")
    
    
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    # ax1.plot(np.array(speeds))
    # ax1.title.set_text('Speeds')
    # #plt.show()
    # ax2.plot(np.array(rewards))
    # ax2.title.set_text('Rewards')
    # plt.show()

        # == Learn from action ==
        #agent.discount_reward()
        # agent.learn(tape)
        
        #r_history.append(score)
        
        

# l = 0
# for key in history:
#     if len(history[key]) > l:
#         l = len(history[key]) 
# for key in history:
#     if len(history[key]) <l:
#         for i in range(int(l-len(history[key]))):
#             history[key].append(None)

df = pd.DataFrame(data=history, columns=["type", "i", "count"])
sns.scatterplot(data=df, x="i", y="count", hue="type")
#sns.lmplot(data=df, x="i", y="count", hue="type")

distances = np.array(distances)
speeds = np.array(speeds)
actions = np.array(actions).reshape(len(actions),2)
rewards = np.array(rewards)
speeds_h = np.array(speeds_h, dtype=object)

print(history)

#df.to_csv(f"history_{datetime.now().strftime('%Y%m%d-%H%M')}.csv")




