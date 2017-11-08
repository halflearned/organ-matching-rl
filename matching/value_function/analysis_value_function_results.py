#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:12:19 2017

@author: vitorhadad
"""

import pickle
import numpy as np
import pandas as pd
from saidman_environment import SaidmanKidneyExchange
from itertools import cycle
from os import listdir


def get_opt_rewards(data, t, h):
    return sum([len(match) for period,match in data["opt_matched"].items()
                if period >= t and period < t + h])

path = "/Users/vitorhadad/Documents/kidney/organ-matching-rl/value_function_results/"

datasets = pickle.load(open("value_function_test_data.pkl","rb"))
    
perf = {}

for file in listdir(path):
    if not file.endswith(".pkl"):
        continue
    
    f = pickle.load(open(path + file, "rb"))
    net = f["net"]
    name = file.split("_")[-1].split(".")[0]
    
    losses = []
    outs = []
    horizon = 10
    n_per_period = 5
    
    for k, data in enumerate(datasets):
    
        
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
            
    print("Name:", name, "Peformance:", np.mean(losses))
    perf[name] = np.mean(losses)
            