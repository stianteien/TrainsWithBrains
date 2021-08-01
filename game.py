# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 20:53:43 2021

@author: Stian
"""

# the game

import pygame
from pygame.locals import *


pygame.init()
width, height = 640, 480
screen=pygame.display.set_mode((width, height))

buss = pygame.image.load("redbus.png")

while 1:
    pygame.event.get() 
    # 5 - clear the screen before drawing it again
    screen.fill(255)
    # 6 - draw the screen elements
    screen.blit(buss, (100,100))
    # 7 - update the screen
    pygame.display.update()
    
    for event in pygame.event.get():
        # check if the event is the X button 
        if event.type==pygame.QUIT:
            # if it is quit the game
            pygame.quit() 
            #exit(0) 