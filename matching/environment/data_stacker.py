#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:40:35 2017

@author: vitorhadad
"""

import numpy as np
from os import listdir
import pickle
from tqdm import trange

from matching.solver.kidney_solver2 import optimal
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.utils.data_utils import get_additional_regressors
from matching.tree_search.mcts import mcts
from matching.utils.data_utils import get_n_matched, clock_seed

files = [f for f in listdir("results/") if f.startswith("optn_")]

f = files[0]

print("Opening data", f)
data  = pickle.load(open("results/" + f, "rb"))
env = data["env"]
o = get_n_matched(data["opt_matched"], env.time_length)
g = get_n_matched(data["greedy_matched"], env.time_length)

maxsize = int(2*env.entry_rate/env.death_rate)
env.removed_container.clear()

As, Xs, ms, ys = [], [], [], []
horizon = 200

for t in trange(env.time_length):
    
    liv = np.array(list(env.get_living(t)))
    n = maxsize - len(liv)
    A = np.pad(env.A(t), ((0,n),(0,n)), mode = "constant", constant_values = 0) 
    X = np.pad(env.X(t), ((0,n),(0,0)), mode = "constant", constant_values = 0)
    opt_m = data["opt_matched"][t]
    y = get_n_matched(optimal(env, t_begin = t, t_end = t+horizon)["matched"])
    
    m = np.zeros(shape = (len(liv),1))
    m[np.isin(liv, list(opt_m))] = 1
    env.removed_container[t].update(opt_m)
    
    As.append(A)
    Xs.append(X)
    ms.append(ms)
    ys.append(y)
    
    
    
    
    
    
    
    
    
    
    






