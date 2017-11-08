#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:58:25 2017

@author: vitorhadad
"""

import numpy as np
import pickle
from make_data import merge_data


for horizon in [29, 44]:

    data = merge_data("X", "N", "G", "y",
                      path = "data/", 
                      horizon = horizon, 
                      max_cycle_length = 2)
    #%%
    Xs,Gs,Ns,ys = [],[],[],[]
    for X,G,N,y in zip(data["N"], data["G"], data["N"], data["y"]):
        if N.shape[1] == 10:
            Xs.append(X)
            Gs.append(G)
            Ns.append(N)
            ys.append(y)
    
    X = np.vstack(Xs)
    G = np.vstack(Gs)
    N = np.vstack(Ns)
    y = np.hstack(ys)
    
    pickle.dump({"X":np.hstack([X,G,N]),"y":y},
            open("training_data_{}.pkl".format(horizon), "wb"))