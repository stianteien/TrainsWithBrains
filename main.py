# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:08:32 2021

@author: Stian
"""

from train import Train
from railway import Railway
from subwaysystem import SubwaySystem
import pandas as pd
import random

linje1coords = pd.read_csv("linje1.csv", index_col=0)
linje2coords = pd.read_csv("linje2.csv", index_col=0)
linje4coords = pd.read_csv("linje4.csv", index_col=0)

linje2stops = pd.concat([linje2coords.sample() ,linje2coords.sample()])
linje2stops["stop_time"] = [500,300] # 10 sekunder
linje2stops["active"] = [False, False]

subwaysystem = SubwaySystem()

# Add all lines
linje1 = Railway(linje1coords)
linje2 = Railway(linje2coords, linje2stops)
linje4 = Railway(linje4coords, train_loop_strategy="line")

# Add all trains
linje1.add_train(Train())
linje2.add_train(Train(direction = -1, max_speed = 300))
linje4.add_train(Train())

# Add lines inn system
subwaysystem.add_railway(linje1)
subwaysystem.add_railway(linje2)
subwaysystem.add_railway(linje4)

subwaysystem.render()