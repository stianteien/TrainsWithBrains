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
        
        self.moves_per_update = self.max_speed * self.update_frequenze
        
    def find_speed(self):
        pass
        
    def accelerate(self, tid):
        
        self.speed = self.max_speed/(1+self.max_speed*np.exp(-self.time_since_stop))
    
    def deaccelarte(self):
        pass