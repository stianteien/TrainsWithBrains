# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:58:46 2021

@author: Stian
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time

import pygame
from pygame.locals import *

class SubwaySystem:
    def __init__(self):
        self.railways = []
        self.frame = np.zeros((600, 500))
        
        
    def add_railway(self, railway):
        self.railways.append(railway)
        
        for info,(x,y) in railway.coords.iterrows():
            self.frame[x,y] = 100
            
    def update(self):
        # Update all trains on all railways
        
        for railway in self.railways:
            for train in railway.trains:
                
                railway.update_train(train)

                                  

    
    def render(self):
        width, height = 600, 500
        screen=pygame.display.set_mode((width, height))
        buss = pygame.image.load("redbus.png")
        surface = pygame.surfarray.make_surface(self.frame)
        #self.buss_coords = (self.railways[0].coords.x[0], self.railways[0].coords.y[0])
        
        # add all train on the grid
        trains = []
        for railway in self.railways:
            for train in railway.trains:
                trains.append((train, pygame.image.load("redbus.png")))
                
        
        while 1:
            pygame.event.get() 
            # 5 - clear the screen before drawing it again
            screen.fill("white")
            # 6 - draw the screen elements
            screen.blit(surface,  (0,0))
            
            # add all trains
            for train, train_bilde in trains:
                screen.blit(train_bilde, train.position)
                
            # 7 - update the screen
            pygame.display.update()
            
            
            # Update all trains and their positons
            self.update()
            

            time.sleep(1e-3)
            
            for event in pygame.event.get():
                
                # check if the event is the X button 
                if event.type==pygame.QUIT:
                    # if it is quit the game
                    pygame.quit() 
                    #exit(0) 