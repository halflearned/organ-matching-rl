#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:33:42 2017

@author: vitorhadad
"""

import numpy as np
from gcn import GraphConvNet, all_sum
from torch import nn, optim
import pickle
from itertools import cycle
from random import choice
from make_data import merge_data


path = "data/"
horizon = 44
max_cycle_length = 2

data = merge_data("X", "N", "G", "opt_obj", "greedy_obj",
                  path = path, 
                  horizon = 44, 
                  max_cycle_length = 2)


X = np.vstack(data["X"])
G = np.vstack(data["G"])
N = np.vstack(data["N"])
ys = [y_opt - y_g for y_opt, y_g in zip(data["opt_obj"], data["greedy_obj"])]
y = np.hstack(ys).T

dadad

sh = np.random.permutation(len(ys))


name = str(np.random.randint(1e8))

losses = []
outs = []

#%%

for k, (A,X,y) in enumerate(cycle(zip(As, Xs, ys))):
            
    sh = np.random.permutation(A.shape[0])    
    loss, out = net.run(A[sh], X[sh], [y])
    
    outs.append(out[0])
        
        
    if k % 100 == 0:
        losses.append(loss) 
        
    if k % 10000 == 0:
        pickle.dump(obj = {"net":net,
                             "entry_rate": data["entry_rate"],
                             "death_rate": data["death_rate"],
                             "training_loss": losses,
                             "num_layers": net.num_layers,
                             "gcn_size": net.gcn_size,
                             "dropout_prob": net.dropout_prob,
                             "horizon": horizon,
                             "n_optimizations": k},
                            file = open("value_function_gcn_{}.pkl"\
                                   .format(name), "wb"))
        
    if k % 101 == 0:
        print(np.mean(losses[-100:]))
        print(out, y)
        

        
        