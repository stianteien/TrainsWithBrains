# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 11:51:11 2022

@author: Stian
"""


from train import Train
from railway import Railway
from subwaysystem import SubwaySystem
import pandas as pd
import numpy as np
import random
from ddqp import DDQNAgent
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
import seaborn as sns


linje10coords = pd.read_csv("lines/linje10.csv", index_col=0)
linje11coords = pd.read_csv("lines/linje11.csv", index_col=0)
linje12coords = pd.read_csv("lines/linje12.csv", index_col=0)

linje10stops = pd.concat([linje10coords[linje10coords.index==80]])
linje10stops[["stop_time", "active"]] = [[100,False]]

linje11stops = pd.concat([linje11coords[linje11coords.index==90]])
linje11stops[["stop_time", "active"]] = [[150,False]]


subwaysystem = SubwaySystem(h=300, w=200)

linje10 = Railway(linje10coords, linje10stops, train_loop_strategy="line")
linje11 = Railway(linje11coords, linje11stops, train_loop_strategy="line")
linje12 = Railway(linje12coords, train_loop_strategy="line")


linje10.add_train(Train())
linje11.add_train(Train())
linje12.add_train(Train())

subwaysystem.add_railway(linje10)
subwaysystem.add_railway(linje11)
subwaysystem.add_railway(linje12)

distances = []#pd.DataFrame([])
for i in range(10):
    subwaysystem.run_simualation()
    distances.append(subwaysystem.distances)

distances = pd.DataFrame(distances).T
j = []
u = []
for i in distances:
    j.extend(distances[i].values)
    u.extend(distances[i].index)
d = pd.DataFrame(np.array([j,u]).T, columns=["distance", "time"])
d["model"] = "Benchmark - manuell stop"
sns.lineplot(data=d, x="time", y="distance", hue="model")
#distances["model"] = "noe"
#distances = pd.DataFrame(np.array(distances).T)

