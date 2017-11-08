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
from kidney_solver import KidneySolver
from os import listdir
from random import shuffle

horizon = 29
max_cycle_length = 2


_, Xs, Ns, Gs = pickle.load(open("training_data.pkl", "rb"))
Xs, Ns, Gs = np.vstack(Xs), np.vstack(Ns), np.vstack(Gs)
sh = np.random.permutation(Xs.shape[0])
Xs, Ns, Gs = Xs[sh], Ns[sh], Gs[sh]

net = GraphConvNet(feature_size = 10,
                   gcn_size = choice([5, 10, 20, 40]),
                   num_layers = choice([1, 2, 3, 4, 5, 10]),
                   dropout_prob = .2,
                   output_fn = all_sum,
                   opt = optim.Adam,
                   opt_params = dict(lr = 0.001),
                   loss_fn = nn.MSELoss)
                   
name = str(np.random.randint(1e8))

losses = []
outs = []

#%%

for k, data in enumerate(cycle(datas)):
            
    A, X = data["A"], data["X"]
    sh = np.random.permutation(A.shape[0])
    y = data["opt_obj"] - data["greedy_obj"]
    
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
                            file = open("value_function_estimates_{}.pkl"\
                                   .format(name), "wb"))
        
    if k % 100 == 0:
        print(np.mean(losses[-100:]))
        

        
        