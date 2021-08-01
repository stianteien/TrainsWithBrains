# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 23:33:47 2021

@author: Stian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import matplotlib.image as mpimg
import matplotlib.animation as animation


import cv2
img = cv2.imread('linje4.png',0)
img[img > 1] = 255

where_pixels = np.where(img == 0)
coords = np.array([(y,x, str(y)+str(x)) for x,y in zip(where_pixels[0], where_pixels[1])])


def give_nearest_coords(coords):
    coords_in_order = [(int(coords[0][0]), int(coords[0][1]))]
    active_coords = coords[1:]
    
    x,y,string_coord = coords[0]
    
    while len(active_coords) > 0:
        #print(len(active_coords))
        
        next_coords = find_nearest(int(x), int(y), active_coords)
        
        if next_coords:
            # pop out x,y in active_coords
            idx = np.where(active_coords[:,2] == string_coord)
            active_coords = np.delete(active_coords, idx[0], axis=0)
            
            x,y,dist,string_coord = next_coords
            coords_in_order.append((int(x),int(y)))
        else:
            coords_in_order.append((int(x),int(y)))
            active_coords = np.delete(active_coords, 0, axis=0)
            
    return coords_in_order
        



def find_nearest(x,y, coords):
    all_distances = []
    for x_, y_,string_coord in coords:
        avstand = np.square((x-int(x_))**2  + (y-int(y_))**2)
        if avstand > 0:
            all_distances.append((x_,y_, avstand, string_coord))
    if len(all_distances) > 0:
        return sorted(all_distances, key=lambda k: k[2])[0]
    

a = give_nearest_coords(coords)
linje1 = pd.DataFrame(a, columns=["x", "y"])
linje1.to_csv("linje4.csv")

    
    
    
    
    