    # -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:05:22 2021

@author: Stian
"""

import numpy as np
import random


class Train:
    def __init__(self, color = "red", max_speed = 100, min_speed = 50,
                 position_index = 0, direction = 1):
        self.color = color
        self.position = None
        self.position_index = position_index
        self.real_position_index = position_index # floating real pos
        self.time_since_stop = 0
        
        self.speed = 0
        self.max_speed = max_speed #moves/seconds
        self.min_speed = min_speed
        self.desired_speed = self.max_speed
        self.desired_action = 1
        
        self.direction = direction # Forward (1) or backwards (-1)
        self.update_frequenze = 1e-3
        self.time_till_stop = 100
        self.distance_to_stopp = 1
        self.been_on_stop = 0
        
        self.start_position = 0
        self.start_index = position_index
        
        self.random_start_index = None
        self.random_start_position = None
        
        
        #self.moves_per_update = self.max_speed * self.update_frequenze
        
    def find_speed(self):
        
        # 3 steps:
        # - Check for stop places
        # - Check for collisons
        # - Change speed
        
        
        #print(self.speed)
        #self.desired_speed_stop_places()
        #self.desired_speed_collisons()
        self.change_speed()
        
        #self.distance_to_stopp += 0.01
        self.moves_per_update = self.speed * self.update_frequenze
        
    def desired_speed_stop_places(self):
        
        x = self.distance_to_stopp # variabel
        k = self.max_speed # constant
        a = .05 #accelration
        
        if self.distance_to_stopp >= 0.2: # -1 before stop and +1 after stop   
            # Slow down
            #self.desired_speed = x
        # Rekker ikke å stoppe!
            self.desired_speed = k*x/(x + k * np.exp(-x*a))
            #self.desired_speed = k*x/(x+k/c*np.exp(-x*a))
            #self.desired_speed = x*1
            
        elif self.distance_to_stopp <= -0.2: # deacs
            #self.desired_speed = -x
            x = -x
            self.desired_speed = k*x/(x + k * np.exp(-x*a))
            #self.desired_speed = k*x/(x+k/c*np.exp(-x))     
        else:
            x = 0
            #self.speed = 0
            self.desired_speed = 0#k*x/(x+k/c*np.exp(-a*x))
            self.been_on_stop += 1
            
        #print(self.desired_speed)
        
        
    def desired_speed_collisons(self):
        
        self.desired_speed = 100 
        
    def speed_up(self):
        accs = 0.3
        speed_change = 0.3
        random_factor = random.random()*0.06
        
        self.speed += speed_change * accs - random_factor
        # Cap speed
        if self.speed >= self.max_speed:
                self.speed = self.max_speed
        
    def speed_down(self):
        accs = 0.3
        speed_change = 0.3
        random_factor = random.random()*0.06
        
        self.speed -= speed_change * accs + random_factor
        # Cap low speed
        if self.speed < 0:
            self.speed = 0
        
        
    def change_speed(self):
        # Desired speed
        # Current speed
        
        # self.desired_speed = 1 SPEED UP
        # slef.desired_speed = 0 SPEED DOWN
        

        if self.desired_action:
            # Speed UP
            self.speed_up()

        else:
            # Speed DOWN
            self.speed_down()
        #else:
        #    self.speed = self.max_speed
        
        '''
        # Det skal varieres med max speed og ikke distanse to stop..!
        if self.desired_speed > self.max_speed:
            self.desired_speed = self.max_speed
        
        current = self.speed
        desired = self.desired_speed
        accs = 0.3
        speed_change = 0.3
        random_factor = random.random()*0.06
        
        #print(f"desired: {self.desired_speed:.2f}, speed: {self.speed:.2f}")
        if np.isclose(self.desired_speed, self.speed, atol= speed_change*accs * 0.9):
            #print(f"Same speed want yeah! current {current:.2f}, desired: {desired:.2f}\
            #      {np.isclose(self.desired_speed, self.speed, 0.2)}")
            self.speed = self.speed
             
        elif self.desired_speed > self.speed:
            # speed up
            #print(f"speeds up! current {current:.2f}, desired: {desired:.2f}")
            self.speed += speed_change * accs - random_factor
            
        elif self.desired_speed < self.speed:
            # speed down
            #print(f"speeds down! current {current:.2f}, desired: {desired:.2f}")
            self.speed -= speed_change * accs + random_factor
            
        '''
            
    
    def deaccelarte(self):
        #x = self. # variabel TIME TILL STOP NOT LESS THAN 0
        # Active a search for see next stop
        
        x = self.time_till_stop
        k = self.max_speed
        if x <= 0: x = 0
            
        if x <= 6:
            if (k*x / (x+k*np.exp(-x))) < k:
                self.speed = k*x/(1*x+np.exp(-x))