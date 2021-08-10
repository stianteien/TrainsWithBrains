# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:05:22 2021

@author: Stian
"""

import numpy as np

class Train:
    def __init__(self, color = "red", max_speed = 100,
                 position_index = 0, direction = 1):
        self.color = color
        self.position = None
        self.position_index = position_index
        self.real_position_index = position_index # floating real pos
        self.time_since_stop = 0
        
        self.speed = 0
        self.max_speed = max_speed #moves/seconds
        
        self.direction = direction # Forward (1) or backwards (-1)
        self.update_frequenze = 1e-3
        self.time_till_stop = 100
        self.distance_to_stopp = 1
        self.been_on_stop = 0
        
        
        #self.moves_per_update = self.max_speed * self.update_frequenze
        
    def find_speed(self):
        
        
        self.accelerate()
        
        #self.distance_to_stopp += 0.01
        self.moves_per_update = self.speed * self.update_frequenze
        
        
    def accelerate(self):
        x = self.distance_to_stopp # variabel
        k = self.max_speed # constant
        c = 5 # Other constat
        a = 0.1 #accelration
        
        if self.distance_to_stopp >= 0.1: # -1 before stop and +1 after stop   
            self.speed = k*x/(x+k/c*np.exp(-a*x))
        elif self.distance_to_stopp <= -0.1: # deacs
            x = -x
            self.speed = k*x/(x+k/c*np.exp(-a*x))     
        else:
            x = 0
            self.speed = k*x/(x+k/c*np.exp(-a*x))
            self.been_on_stop += 1
            
    
    def deaccelarte(self):
        #x = self. # variabel TIME TILL STOP NOT LESS THAN 0
        # Active a search for see next stop
        
        x = self.time_till_stop
        k = self.max_speed
        if x <= 0: x = 0
            
        if x <= 6:
            if (k*x / (x+k*np.exp(-x))) < k:
                self.speed = k*x/(1*x+np.exp(-x))