# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:06:58 2021

@author: Stian
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time

import pygame
from pygame.locals import *

class Railway:
    def __init__(self, coords):
        self.coords = coords
        self.trains = []
        self.start = self.coords.loc[0]
        self.end = self.coords.loc[len(self.coords)-1]
        self.end_index = len(self.coords)-1
        self.start_index = 0
        
        self.train_loop = "loop"
        #self.end_strategi = self.start
        
        
        
        # self.frame = np.zeros((300, 500))
        # for info,(x,y) in self.coords.iterrows():
        #     self.frame[x,y] = 1
        
        #self.buss = cv.imread("buss.png")[:,:,0]
        #self.buss[self.buss < 200] = 0
        
        #self.buss_coords = (self.coords.x[0], self.coords.y[0])
        
    def update_train(self, train):
        update_speed = 1
        
        
        if not self.end_strategi(train):
            # end not react - do update
            
            update_speed = train.moves_per_update
            train.real_position_index += update_speed * train.direction
            
            train.position_index = int(np.floor(train.real_position_index))
            train.position = (self.coords.x[train.position_index],
                              self.coords.y[train.position_index])
            
            
        
        
        
    def end_strategi(self, train):
        # For loop-strategy
        if self.train_loop == "loop":
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
                
                
        
    def add_train(self, train):
        self.trains.append(train)
        train.position_index = train.position_index
        train.position = (self.coords.x[train.position_index],
                          self.coords.y[train.position_index])
        

       