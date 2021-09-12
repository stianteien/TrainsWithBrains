# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:05:22 2021

@author: Stian
"""

import numpy as np
import random


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
        self.desired_speed = self.max_speed
        
        self.direction = direction # Forward (1) or backwards (-1)
        self.update_frequenze = 1e-3
        self.time_till_stop = 100
        self.distance_to_stopp = 1
        self.been_on_stop = 0
        
        
        
        
        #self.moves_per_update = self.max_speed * self.update_frequenze
        
    def find_speed(self):
        
        # 3 steps:
        # - Check for stop places
        # - Check for collisons
        # - Change speed
        
        
        
        self.desired_speed_stop_places()
        #self.desired_speed_collisons()
        self.change_speed()
        
        #self.distance_to_stopp += 0.01
        self.moves_per_update = self.speed * self.update_frequenze
        
    def desired_speed_stop_places(self):
        
        x = self.distance_to_stopp # variabel
        k = self.max_speed # constant
        c = 5 # Other constat
        a = 1 #accelration
        
        if self.distance_to_stopp >= 0.2: # -1 before stop and +1 after stop   
            # Slow down
            #self.desired_speed = x
        
            self.desired_speed = k*x/(x+k/c*np.exp(-x))
            
        elif self.distance_to_stopp <= -0.2: # deacs
            #self.desired_speed = -x
            x = -x
            self.desired_speed = k*x/(x+k/c*np.exp(-x))     
        else:
            x = 0
            self.speed = 0
            self.desired_speed = 0#k*x/(x+k/c*np.exp(-a*x))
            self.been_on_stop += 1
        
        
    def desired_speed_collisons(self):
        
        self.desired_speed = 100 
        
    def change_speed(self):
        # Desired speed
        # Current speed
        
        # Det skal varieres med max speed og ikke distanse to stop..!
        
        current = self.speed
        desired = self.desired_speed
        accs = 1
        random_factor = random.random()*0.06
        
        
        if desired > current:
            # speed up
            self.speed += 0.3 * accs - random_factor
            
        elif desired < current:
            # speed down
            self.speed -= 0.3 * accs + random_factor
            
        elif np.isclose(desired, current, 0.2):
             self.speed = self.speed
        
            
    
    def deaccelarte(self):
        #x = self. # variabel TIME TILL STOP NOT LESS THAN 0
        # Active a search for see next stop
        
        x = self.time_till_stop
        k = self.max_speed
        if x <= 0: x = 0
            
        if x <= 6:
            if (k*x / (x+k*np.exp(-x))) < k:
                self.speed = k*x/(1*x+np.exp(-x))