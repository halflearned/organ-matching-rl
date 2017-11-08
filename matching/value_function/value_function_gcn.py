#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:33:42 2017

@author: vitorhadad
"""

import numpy as np
from torch import nn, optim
import pickle
from itertools import cycle
from sys import argv

from matching.utils.data_utils import merge_data
from matching.gcn.gcn import GraphConvNet, all_sum

if len(argv) == 1:
    gcn_size = 1
    num_layers = 1
    horizon = 22
else:
    gcn_size = int(argv[1])
    num_layers = int(argv[2])
    horizon = int(argv[3])


#%% Get data
datasets = merge_data("X", "t", "A", "G", "N", "opt_n_matched", "greedy_n_matched", path = "data/")
X_shape = datasets[0][0]["X"].shape[1]
N_shape = datasets[0][0]["N"].shape[1]
G_shape = datasets[0][0]["G"].shape[1]


#%%
# Flattening
n = 195
XXs = []

t_begin = 90
t_end = t_begin + horizon
As = []
ts = []
ys = []
gs = []
for ds in datasets:
    for item in ds:
        t = item["t"]
        if t == t_begin:
            As.append(item["A"].toarray())
            XXs.append(np.hstack([item["X"], item["G"], item["N"]]))
            ys.append(np.sum(item["opt_n_matched"][t_begin:t_end]))
            gs.append(np.sum(item["greedy_n_matched"][t_begin:t_end]))
            

training_data = cycle(zip(As, XXs, ys))
#%%
    

net = GraphConvNet(feature_size = X_shape + N_shape + G_shape,
                   gcn_size = gcn_size,
                   num_layers = num_layers,
                   dropout_prob = .2,
                   output_fn = all_sum,
                   opt = optim.Adam,
                   opt_params = dict(lr = 0.001),
                   loss_fn = nn.MSELoss)
                   

name = str(np.random.randint(1e8))

training_loss = []
k = 0
#%%

while True:
    
    A, XX, y = next(training_data)
    
    loss, out = net.run(A, XX, [y])
    
        
    if k % 100 == 0:
        training_loss.append(loss)
        
    if k % 1000 == 0:
        print("\n\nLoss: ", np.mean(training_loss[-100:]))
        print("Estimate: ", out[0])
        print("Truth: ", y)
        
        
    if k % 10000 == 0:
        pickle.dump(obj = {"net":net,
                         "training_loss": training_loss,
                         "num_layers": net.num_layers,
                         "gcn_size": net.gcn_size,
                         "dropout_prob": net.dropout_prob,
                         "horizon": horizon},
                        file = open("value_function_gcn_{}_{}_{}.pkl"\
                                   .format(gcn_size, num_layers, horizon), "wb"))
        
    k += 1