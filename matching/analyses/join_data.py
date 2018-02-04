#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:40:54 2018

@author: halflearned
"""

import numpy as np
from glob import glob
from sys import argv

env = argv[1]

data_x = []
data_y = []
data_g = []
data_e = []

for f in sorted(glob("data/X_{}*npy".format(env))):
    print(f)
    seed = f.split("_")[-1].split(".")[0]
    try:
        X = np.load("data/X_{}_{}.npy".format(env, seed))
        Y = np.load("data/Y_{}_{}.npy".format(env, seed))
        G = np.load("data/G_{}_{}.npy".format(env, seed))
        E = np.load("data/E_{}_{}.npy".format(env, seed))
        data_x.append(X)
        data_y.append(Y)
        data_g.append(G)
        data_e.append(E)
    except:
        print(seed)    

X = np.vstack(data_x)
G = np.vstack(data_g)
E = np.vstack(data_e)
Y = np.hstack(data_y)

assert X.shape[0] == G.shape[0] == E.shape[0] == Y.shape[0]

np.save("data/X_{}.npy".format(env), X)
np.save("data/G_{}.npy".format(env), G)
np.save("data/E_{}.npy".format(env), E)
np.save("data/Y_{}.npy".format(env), Y)