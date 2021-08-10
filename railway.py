# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:06:58 2021

@author: Stian
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time
import pandas as pd

import pygame
from pygame.locals import *

class Railway:
    def __init__(self, coords, stop_places=pd.DataFrame([]), 
                 train_loop_strategy="loop"):
        self.coords = coords
        self.trains = []
        self.start = self.coords.loc[0]
        self.end = self.coords.loc[len(self.coords)-1]
        self.end_index = len(self.coords)-1
        self.start_index = 0
        
        self.stop_places = stop_places # Indexes as df
        
        self.train_loop_strategy = train_loop_strategy

        
    def update_train(self, train):
        update_speed = 1
        
        
        if not self.end_strategi(train):
            # end not react - do update
            
            if not self.stop_places.empty:
                self.give_stop_distances(train)
                
                # Start the train
                if train.been_on_stop >= self.current_stop_place.stop_time:
                    train.been_on_stop = 0
                    train.real_position_index += 0.2 * train.direction
            else:
                train.distance_to_stopp += 0.01
            train.find_speed()
            update_speed = train.moves_per_update
            train.real_position_index += update_speed * train.direction
            
            #if train.real_position_index > 0: train.real_position_index = 0
            train.position_index = int(np.round(train.real_position_index)) #kan klikke ettervert
            train.position = (self.coords.x[train.position_index],
                              self.coords.y[train.position_index])
            
            
        
    def give_stop_distances(self, train):
        
        
        b = np.array(train.position)
        a = np.array([self.stop_places.x.tolist(), self.stop_places.y.tolist()]).T
        self.stop_places["distance"] = np.linalg.norm(a-b, axis=1)
        
        #b = train.real_position_index #pos of train 66.6    
        a = self.stop_places.distance.tolist() # steder å stoppe df with  x[10,70,100] y -.-
        

        idx = np.abs(a).argmin()
        nearest = a[idx]
        self.current_stop_place = self.stop_places.iloc[idx].copy()
        avstand = nearest
        
        avstand = avstand *  train.direction # Hvilken vei den kjører 1
        
        train.distance_to_stopp = avstand
        
        if train.distance_to_stopp <= 0.1 and train.distance_to_stopp >= -0.1:
            actives = [False for i in range(len(self.stop_places))]
            actives[idx] = True
            self.stop_places.active = actives
        else:
            self.stop_places.active = False
        # return closest stop, (-) if behind, (+) if ahead
        
        #train.stop_distances
        ##train.time_till_stop # Skal forandres
        
        
    def end_strategi(self, train):
        # For loop-strategy
        if self.train_loop_strategy == "loop":
            if train.direction == 1: #  1 forward
                # Check if end is end and return start
                if train.real_position_index >= self.end_index:
                    train.position = self.start
                    train.real_position_index = self.start_index
                    train.position_index = self.start_index
                    return 1
                
            else: # -1 backwards
                # Check if end is start and return end
                if train.position_index <= self.start_index:
                    train.position = self.end
                    train.real_position_index = self.end_index
                    train.position_index = self.end_index
                    return 1
            
        # for line strategy
        if self.train_loop_strategy == "line":
            if train.real_position_index > self.end_index \
                    or train.real_position_index < self.start_index: 
                
                if train.direction == 1:
                    train.direction = -1
                else:
                    train.direction = 1
                
                
        
    def add_train(self, train):
        self.trains.append(train)
        train.position_index = train.position_index
        train.position = (self.coords.x[train.position_index],
                          self.coords.y[train.position_index])
        

       