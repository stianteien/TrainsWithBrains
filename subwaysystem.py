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
        self.done = False
        
        
    def add_railway(self, railway):
        self.railways.append(railway)
        self.frame[railway.coords.x, railway.coords.y] = 100
            
    def find_intersections(self):
        # Find for all stops that are same
        pass
        
        # Find for all lines that intersect
        #for railway in self.railways:
            
        
        
            
    def update(self):
        # Update all trains on all railways
        
        for railway in self.railways:
            for train in railway.trains:
                railway.update_train(train)
                
    def run_simualation(self):
        while not self.done:
            self.update()
            time.sleep(1e-3)
        

                                  

    
    def render(self):
        pygame.font.init()
        width, height = 600, 500
        screen=pygame.display.set_mode((width, height))
        buss = pygame.image.load("redbus.png")
        surface = pygame.surfarray.make_surface(self.frame)
       
        # add all train on the grid
        trains = []
        for railway in self.railways:
            for train in railway.trains:
                trains.append((train, 
                               pygame.font.SysFont('Comic Sans MS', 13),
                               pygame.image.load("redbus.png")))
                
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
                
        
                
        
        while not self.done:
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
                        stop_bilde = pygame.image.load("stop_place_red.png")
                    else:
                        stop_bilde = pygame.image.load("stop_place.png") 
                    screen.blit(stop_bilde,
                                (x,y-38))
            
            # add all trains
            for train, tekst, train_bilde in trains:
                screen.blit(train_bilde, train.position)
                screen.blit(tekst.render(str(np.round(train.speed,0))+" m/s", False, (255, 255, 255)), 
                            (train.position[0], train.position[1]-15))
            
                
                
            # 7 - update the screen
            pygame.display.update()
            
            
            # Update all trains and their positons
            self.update()
            

            #time.sleep(1e-3)
            
            for event in pygame.event.get():
                
                # check if the event is the X button 
                if event.type==pygame.QUIT:
                    # if it is quit the game
                    pygame.quit() 
                    #exit(0) 