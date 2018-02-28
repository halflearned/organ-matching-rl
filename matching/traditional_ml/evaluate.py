#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:27:19 2018

@author: vitorhadad
"""

from sys import platform, argv
from tqdm import trange

import numpy as np
from random import choice
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from matching.solver.kidney_solver2 import optimal, greedy
from matching.utils.data_utils import clock_seed
from matching.environment.abo_environment import ABOKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.utils.data_utils import get_additional_regressors, run_node2vec, get_n_matched
from sklearn.externals import joblib
    

if platform=="darwin":
    argv = [None, "abo", "lr", "none", 5, 0.1]

envtype = argv[1]
algorithm = argv[2]
add = argv[3]
entry_rate = float(argv[4])
death_rate = float(argv[5])/100
thres = 0.3
algo = joblib.load("results/traditional_ml/{}_{}_{}.pkl".format(envtype, algorithm, add))

seed = clock_seed()

args = {'entry_rate': entry_rate,
        'death_rate': death_rate,
        'time_length': 1001,
        'seed': seed}

if envtype == "abo":
    env = ABOKidneyExchange(,
elif envtype == "saidman":
    env = SaidmanKidneyExchange(,
elif envtype == "optn":
    env = OPTNKidneyExchange(**args)

opt = optimal(env)
gre = greedy(env)

g = get_n_matched(gre["matched"], 0, env.time_length)
o = get_n_matched(opt["matched"], 0, env.time_length)

#%%
rewards = np.zeros(env.time_length)
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
    elif add == "both":
        E = run_node2vec(A[has_cycle, :][:, has_cycle])
        G = get_additional_regressors(env, t, dtype="numpy")[has_cycle]
        XX = np.hstack([X, G, E])

    yhat = algo.predict_proba(XX)[:,1] >  thres
    yhat_full[has_cycle] = yhat
    potential = liv[yhat_full]
    
    removed = optimal(env, t, t, subset=potential)["matched"][t]
    env.removed_container[t].update(removed)
    rewards[t] = len(removed)


    stats = [envtype,
             algorithm,
             thres,
             add,
             seed,
             t,
             int(env.entry_rate),
             int(env.death_rate * 100),
             rewards[t],
             g[t],
             o[t]]
    msg = ",".join(["{}"] * len(stats)).format(*stats)

    if platform == "linux":
        with open("results/traditional_ml_mdp_results2.txt", "a") as f:
            print(msg, file=f)
    else:
        print(stats)



    