#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:33:42 2017

@author: vitorhadad
"""

from saidman_environment import SaidmanKidneyExchange
import numpy as np
from gcn import GraphConvNet, all_sum
from torch import nn, optim
import pickle
from itertools import cycle
from random import choice


def get_opt_rewards(data, t, h):
    return sum([len(match) for period,match in data["opt_matched"].items()
                if period >= t and period < t + h])
    
horizon = 10
n_per_period = 10

net = GraphConvNet(feature_size = 10,
                   gcn_size = choice([5, 10, 20, 50, 100]),
                   num_layers = choice([1, 3, 5, 10, 20]),
                   dropout_prob = .2,
                   output_fn = all_sum,
                   opt = optim.Adam,
                   opt_params = dict(lr = 0.001),
                   loss_fn = nn.MSELoss)
                   
name = str(np.random.randint(1e8))
datasets = pickle.load(open("predict_obj_data.pkl","rb"))

losses = []
outs = []

#%%

#for data in cycle(datasets):
for k, data in enumerate(cycle(datasets)):

    
    random_times = np.random.randint(data["time_length"] - horizon - 1, 
                                     size = n_per_period)
    

    
    env = SaidmanKidneyExchange(entry_rate  = data["entry_rate"],
                                death_rate  = data["death_rate"],
                                time_length = data["time_length"],
                                seed = data["seed"])
    
    for t in random_times:
    
        A, X = env.A(t), env.X(t)
        n = A.shape[0]
        sh = np.random.permutation(n)
        
        y = get_opt_rewards(data, t, horizon)
        
        loss, out = net.run(A[sh], X[sh], [y])
        losses.append(loss) 
        outs.append(out[0])
        
        
    if k % 1000 == 0:
        print("Iteration:", k)
        print(np.mean(losses[-1000:]))    
        pickle.dump({"net":net,
                     "entry_rate": data["entry_rate"],
                     "death_rate": data["death_rate"],
                     "training_loss": losses},
                     open("net_predict_obj_{}.pkl".format(name), "wb"))
        
#%%
pickle.load()
        
        
        