#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:33:52 2017

@author: vitorhadad
"""

from copy import deepcopy
import numpy as np
import pandas as pd
from random import shuffle, choice
from collections import Counter
import pickle
from sys import argv, platform

from matching.environment.optn_environment import OPTNKidneyExchange
from matching.solver.kidney_solver2 import optimal_with_discount, optimal, greedy
from matching.utils.data_utils import  clock_seed, cumavg
from matching.utils.env_utils import snapshot, remove_taken, get_loss
from matching.utils.data_utils import flatten_matched, disc_mean , get_n_matched
#%%


def greedy_actions(env, t, n_repeats):
    return set([tuple(optimal(env, t, t)["matched"][t]) 
                for _ in range(n_repeats)])
    
def simulate_value(snap, t, horizon, gamma, n_iters):
    values = []
    for i in range(n_iters):
        snap.populate(t+1, t+horizon+1, seed = clock_seed())  
        opt = optimal_with_discount(snap, t, t+horizon, gamma = gamma)
        values.append(opt["obj"])
    return values
    

def predict(env, t, horizon, gamma, n_iters, n_repeats = 20):
    g_acts = greedy_actions(env, t, n_repeats)
    if len(g_acts) == 0:
        return 0
    values = dict()
    for g in g_acts:
        snap = snapshot(env, t)
        snap.removed_container[t].update(g_acts)
        avg_value = np.mean(simulate_value(snap, t, horizon, 
                                       gamma, n_iters))
        values[g] = avg_value
    return values


#%%

env = OPTNKidneyExchange(5, .1, 2000, seed = 12345)

opt = optimal(env)
gre = greedy(env)
o = get_n_matched(opt["matched"], 0, env.time_length)
g = get_n_matched(gre["matched"], 0, env.time_length)

#%%
time_limit = 100 #env.time_length
t = 0
if platform == "linux":
    horizon = int(argv[1]) 
    n_iters = int(argv[2])
    gamma = float(argv[3])
else:
    horizon = 50 #int(argv[1]) 
    n_iters = 5 # int(argv[2])
    gamma = .9# float(argv[3])
n_times = 20
rewards = []
matched = set()       
rndname = str(np.random.randint(1e8))
outfilename = "quicksearch_{}_{}_{}_{}".format(horizon, 
               n_times, gamma, rndname)


if platform == "linux":
    outfile = open(outfilename + ".txt", "a")
else:
    outfile = None
#%%

while t < time_limit:

    c = predict(env, t, horizon,  gamma = gamma, n_iters = n_iters)
    a = max(c)
    env.removed_container[t].update(a)
    rewards.append(len(a))
    
    print(np.mean(rewards), 
          np.mean(g[:t+1]),
          np.mean(o[:t+1]),
          file = outfile)
    
    t += 1
    
    with open(outfilename + ".pkl", "wb") as f:
        pickle.dump(obj = {"rewards":rewards,
                     "o":o[1:t+1],
                     "g":g[1:t+1]},
                    file = f)
 
if platform == "linux":
    outfile.close()
#%%
#plt.plot(cumavg(rewards), linewidth = 2);
#plt.plot(cumavg(g[1:t+1]));
#plt.plot(cumavg(o[1:t+1]));
#plt.ylim(3.5, 5)
    
    

    
    
    


