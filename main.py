# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:08:32 2021

@author: Stian
"""

from train import Train
from railway import Railway
from subwaysystem import SubwaySystem
import pandas as pd

linje1coords = pd.read_csv("linje1.csv", index_col=0)
linje2coords = pd.read_csv("linje2.csv", index_col=0)
linje3coords = pd.read_csv("linje3.csv", index_col=0)

subwaysystem = SubwaySystem()

# Add all lines
linje1 = Railway(linje1coords)
linje2 = Railway(linje2coords)
linje3 = Railway(linje3coords)

# Add all trains
linje1.add_train(Train())
linje2.add_train(Train(direction = -1, max_speed = 300))
linje3.add_train(Train())

# Add lines inn system
subwaysystem.add_railway(linje1)
subwaysystem.add_railway(linje2)
subwaysystem.add_railway(linje3)

subwaysystem.render()