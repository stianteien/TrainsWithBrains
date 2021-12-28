# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:58:46 2021

@author: Stian
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv
import time
import collections
import random
from scipy.spatial.distance import pdist, squareform


import pygame
from pygame.locals import *

class SubwaySystem:
    def __init__(self):
        self.railways = []
        self.frame = np.zeros((300, 200))
        self.done = False
        self.state = None
        self.info = None
        self.reward = None
        
        self.action_history = []
        self.counter = 0 
        
    def add_railway(self, railway):
        self.railways.append(railway)
        self.frame[railway.coords.x, railway.coords.y] = 100
            
    def find_intersections(self):
        # Find for all stops that are same
        df = pd.concat([railway.coords for railway in self.railways])
        df = df.reset_index(drop=True)
        df_gpby = df.groupby(list(df.columns))
        df_same = df_gpby.size()
        df_same = df_same[df_same > 1]
        df_same = np.array([[x,y] for x,y in df_same.index.to_numpy()])
        df_same = pd.DataFrame(df_same, columns=["x","y"])
        
        # set stop sign @ their places
        
        #self.frame[df_same.x, df_same.y] = 1000
        #plt.imshow(self.frame, cmap="hot")
        #print(df_same)
        
    def check_for_crash(self, trains):
        # Check if trains is on the same position
        distances = pdist([train.position for train, _,_ in trains])
                     
        if np.sort(distances)[0] < 10:
            self.done = True
            print("Kræsj")

            
            
    def update(self, actions):
        # Update all trains on all railways
        for (train, _,_) , action in zip(self.trains, actions[0]):
            train.desired_action = action
        
        for railway in self.railways:
            for train in railway.trains:
                railway.update_train(train)
                
        #print([train.speed for train, _,_ in self.trains])
                
    def run_simualation(self):
        trains = []
        o = 0
        self.reset()
        
                
        while not self.done:
            o +=1    
            #self.step(action)
            self.check_for_crash(self.trains)            
            self.update()  
            
            if o % 1000 == 0:
                print(o)
    
    def reset(self):
        self.trains = []
        self.action_history = []
        o = 0
        for railway in self.railways:
            for train in railway.trains:
                self.trains.append((train, 0, 0))
                
        for train, _, _ in self.trains:
            index, train.position_index = random.choice(list(enumerate(train.random_start_index)))
            train.position = train.random_start_position[index]
            train.real_position_index = train.position_index
            train.speed = 0
            
        self.state = np.array([])
        for train,_,_ in self.trains:
            self.state = np.append(self.state, [train.speed,
                                                train.real_position_index,
                                                train.desired_action])
            distances = pdist([train.position for train, _,_ in self.trains])
            self.state = np.append(self.state, distances)
                
        self.reward = 0
        self.counter = 0 
        self.done = False
        return self.state, self.done
    
    
    def step(self, actions):
        '''
        Parameters
        ----------
        action : speed of all trains?
            Action to be taken.

        Returns
        -------
        state
            the state of the enviroment
        reward
            reward of from this state.
        done
            if env is done.
        info
            extra information.
        '''
        #action - list of desired speed of all trains!
        #if len(self.action_history)>50:
        #    if (self.action_history[-1] == np.mean(self.action_history[-50:], axis=0)).all():
        #        action = np.array([100, 100])*np.random.rand(1,2)[0]
        
        
        self.check_for_crash(self.trains)            
        self.update(actions)

        #self.action_history.append(action)
        #self.action_history = np.append(self.action_history, action)
        
        # =====
        # Return all state information
        # =====
        
        # Gi tilbake posisjon for toget i koords og fart til toget? (1,3)?
        # Tog - (korrds, fart, avstand til andre?)
        # (koordx1, koordy1, fart1, avstand12, koordx2, ... osv)
        # -->Reward kan bli fart + avstand
        self.reward = 0
        self.state = np.array([])
        for train,_,_ in self.trains:
            self.state = np.append(self.state, [train.speed,
                                                train.real_position_index,
                                                train.desired_speed])
            distances = pdist([train.position for train, _,_ in self.trains])
            self.state = np.append(self.state, distances)
            
            self.reward += train.speed + distances
        
        self.counter += 1
        #self.reward = self.reward - self.counter
        
        return self.state, self.reward/100, self.done, self.info

                                  

    
    def render(self, agent):
        pygame.font.init()
        width, height = 300, 200
        screen=pygame.display.set_mode((width, height))
        buss = pygame.image.load("img/redbus.png")
        surface = pygame.surfarray.make_surface(self.frame)
       
        # add all train on the grid
        trains = []
        for railway in self.railways:
            for train in railway.trains:
                trains.append((train, 
                               pygame.font.SysFont('Comic Sans MS', 13),
                               pygame.image.load("img/redbus.png")))
                
        # Add all stop places on grid
        # stop_places = []
        # for railway in self.railways:
        #     for i, (x,y,stop_time, active) in railway.stop_places.iterrows():
        #         if active:
        #             stop_bilde = pygame.image.load("stop_place_red.png")
        #         else:
        #             stop_bilde = pygame.image.load("stop_place.png")  
        #         stop_places.append((stop_bilde,
        #                            (x,y-38), active))
                
        state, done = self.reset()
                
        
        while True:
            
            if not self.done:
                # This needs to be updated all the time.
                pygame.event.get() 
                # 5 - clear the screen before drawing it again
                screen.fill("white")
                # 6 - draw the screen elements
                screen.blit(surface,  (0,0))
                
                # add all stopplaces
                for railway in self.railways:
                    for i, (x,y,stop_time, active) in railway.stop_places.iterrows():
    
                        # Prøv å blit kun ved forandring
                        if active:
                            stop_bilde = pygame.image.load("img/stop_place_red.png")
                        else:
                            stop_bilde = pygame.image.load("img/stop_place.png") 
                        screen.blit(stop_bilde,
                                    (x,y-38))
                
                # add all trains
                for train, tekst, train_bilde in trains:
                    screen.blit(train_bilde, train.position)
                    screen.blit(tekst.render(str(np.round(train.speed,0))+" km/h", False, (255, 255, 255)), 
                                (train.position[0], train.position[1]-15))
                
                    
                    
                # Check for collisons
                self.check_for_crash(trains)
                
    
                
                # 7 - update the screen
                pygame.display.update()
                
                
                # 8 - Update all trains and their positons
                action = agent.choose_action(state)
                state_, reward, done, info = self.step(action)
                state = state_
            
            

            #time.sleep(1e-3)
            
            for event in pygame.event.get():
                
                # check if the event is the X button 
                if event.type==pygame.QUIT:
                    # if it is quit the game
                    pygame.quit() 
                    #exit(0) 