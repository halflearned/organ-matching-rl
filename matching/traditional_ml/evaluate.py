#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:27:19 2018

@author: vitorhadad
"""


from sys import platform, argv
from tqdm import trange

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from matching.solver.kidney_solver2 import optimal, greedy
from matching.utils.data_utils import clock_seed
from matching.environment.abo_environment import ABOKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.utils.data_utils import get_additional_regressors, run_node2vec, get_n_matched
from sklearn.externals import joblib
    

if platform=="darwin":
    argv = [None, "lr", "abo", "none", .5]


algo = joblib.load(argv[1])
envtype = argv[2]
add = argv[3]
thres = argv[4]

args = dict(entry_rate = 5, death_rate = .1, 
         time_length = 2000, seed = clock_seed())

if envtype == "abo":
    env = ABOKidneyExchange(**args)
elif envtype == "saidman":
    env = SaidmanKidneyExchange(**args)
elif envtype == "optn":
    env = OPTNKidneyExchange(**args)

opt = optimal(env)
gre = greedy(env)

#%%
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
    if add == "none":
        XX = X
    elif add == "networkx":
        G = get_additional_regressors(env, t, dtype="numpy")[has_cycle] 
        XX = np.hstack([X, G])
    elif add == "node2vec":
        E = run_node2vec(A[has_cycle,:][:,has_cycle])
        XX = np.hstack([X, E])
        
    yhat = algo.predict_proba(XX)[:,1] > thres
    yhat_full[has_cycle] = yhat
    potential = liv[yhat_full]
    
    removed = optimal(env, t, t, subset=potential)["matched"][t]
    env.removed_container[t].update(removed)
    rewards.append(len(removed))



gre_n = get_n_matched(gre["matched"], 0, env.time_length)
opt_n = get_n_matched(opt["matched"], 0, env.time_length)

print("\nrewards\n",
      np.sum(rewards),
      np.sum(gre_n),
      np.sum(opt_n))
    