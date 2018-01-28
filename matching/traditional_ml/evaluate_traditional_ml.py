#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:27:19 2018

@author: vitorhadad
"""


from sys import platform
from random import choice

import numpy as np
import pickle

from matching.utils.data_utils import clock_seed
from matching.environment.abo_environment import ABOKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.utils.data_utils import get_features
    



env = ABOKidneyExchange(entry_rate = 5,
                        death_rate = .1, 
                        time_length = 1500, 
                        seed = clock_seed())

opt = optimal(env)
gre = greedy(env)

#%%
def evaluate(algo, env, thres):

    env.removed_container.clear()
    rewards = []
    for t in trange(env.time_length):
        
        liv = np.array(env.get_living(t))
        A = env.A(t)
        has_cycle = np.diag(A @ A) > 0
        liv_and_cycle = liv[has_cycle]
        yhat_full = np.zeros(len(liv), dtype=bool)
        
        if len(liv_and_cycle) == 0:
            continue
        
        X = env.X(t)[has_cycle]
        subg = env.subgraph(liv_and_cycle)
        E = run_node2vec(subg) 
        F = np.hstack([X, E])
    
        yhat = algo.predict_proba(F)[:,1] > thres
        yhat_full[has_cycle] = yhat
        potential = liv[yhat_full]
        
        removed = optimal(env, t, t, subset=potential)["matched"][t]
        env.removed_container[t].update(removed)
        rewards.append(len(removed))
        
    return rewards


r = evaluate(pipe, env, .05)

gre_n = get_n_matched(gre["matched"], 0, env.time_length)
opt_n = get_n_matched(opt["matched"], 0, env.time_length)

print("\nrewards\n",
      np.sum(r[500:]),
      np.sum(gre_n[500:]),
      np.sum(opt_n[500:]))
    