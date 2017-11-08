#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:50:05 2017

@author: vitorhadad
"""


import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from collections import defaultdict
from itertools import chain

from matching.gcn.policy_function_gcn import mdp
from matching.solver.kidney_solver import KidneySolver
from matching.environment.saidman_environment import SaidmanKidneyExchange

path= "policy_function_results/"
thresholds = [.1, .25, .5] #np.linspace(0.05, 0.95, 19)
 
tabs = []

files = listdir(path)
for f in files:
    
    if f.startswith("policy_gcn_"):
        
        name = path + f
        try:
            r = pickle.load(open(name, "rb"))
        except EOFError:
            print("Error when loading", f)
        
            
        net = r["net"]

        net.eval()
        test_perf = defaultdict(list)
    
        print("file", f)
        print("gcn_size", net.gcn_size)
        print("num_layers", net.num_layers)
        print("thres", thres)
    
        for thres in thresholds:

            env = SaidmanKidneyExchange(entry_rate  = 5, #choice([2, 5, 10]),
                                    death_rate  = .1, #choice([0.1, 0.01, 0.005]),
                                    time_length = 100,
                                    seed = 0)
            
            
            solver = KidneySolver(max_cycle_length = 2, #choice([2,3]),
                                  burn_in = 0)  
            
#            opt = solver.optimal(env)
#            greedy = solver.greedy(env)
            
            rs = []
            ms = []
            for t in range(env.time_length):
                
                idx = np.array(env.get_living(t))
                A = env.A(t)
                X = env.X(t)
                probs = net.forward(A, X).data.numpy() 
                chosen = idx[probs >= thres]
                sol = solver.solve_subset(env, chosen)
                m = list(chain(*sol["matched"].values()))
                rs.append(len(m))
                ms.extend(m)
                env.removed_container[t].update(m)
                

            print(len(ms))