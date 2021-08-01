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
        
        
        #self.moves_per_update = self.max_speed * self.update_frequenze
        
    def find_speed(self):
        self.accelerate()
        self.time_since_stop += 0.01
        self.moves_per_update = self.speed * self.update_frequenze
        
    def accelerate(self):
        x = self.time_since_stop # variabel
        k = self.max_speed # constant
        self.speed = k*x/(1*x+k*np.exp(-x))
    
    def deaccelarte(self):
        #x = self. # variabel TIME TILL STOP NOT LESS THAN 0
        x = 0
        if x <= 0: x = 0
        k = self.max_speed # constant
        self.speed = k*x/(1*x+k*np.exp(-x))
        
        