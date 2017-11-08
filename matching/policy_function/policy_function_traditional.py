#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 20:33:42 2017

@author: vitorhadad
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from make_data import undersample, oversample
from sys import argv
from saidman_environment import SaidmanKidneyExchange
from kidney_solver import KidneySolver
from make_data import get_additional_regressors
from itertools import chain



if len(argv) > 1:
    horizon = int(argv[1]) # 29 or 44
    max_cycle_length = int(argv[2]) # 2 only for now
    algo_number = int(argv[3]) # one out of 44
    sampling = int(argv[4]) # one out of 3
else:
    horizon = 29 # 29 or 44
    max_cycle_length = 2 # 2 only for now
    algo_number = 0 # one out of [0,...,43]
    sampling = 0 # one out of [0,1,2]


for i,a in enumerate(argv):
    print("arg",i,":",a)


data = pickle.load(open("training_data_{}.pkl".format(horizon),"rb"))
XX, yy = data["X"], data["y"]

if sampling == 1:
    XX, yy = undersample(XX, yy)  
elif sampling == 2:
    XX, yy = oversample(XX, yy) 
    
    
name = str(np.random.randint(1e8))
    
#%%
thresholds = np.linspace(0.05, .95, 19)

algos = [LogisticRegression(penalty = "l1", C = 100),
        LogisticRegression(penalty = "l1", C = 1),
        LogisticRegression(penalty = "l1", C = .01),
        LogisticRegression(penalty = "l2", C = 100),
        LogisticRegression(penalty = "l2", C = 1),
        LogisticRegression(penalty = "l2", C = .01),
        SVC(C = .1),
        SVC(C = 1),
        SVC(C = 10),
        SVC(C = 1000),
        SVC(C = 10000),
        GaussianProcessClassifier(),
        RandomForestClassifier(n_estimators = 100),
        RandomForestClassifier(n_estimators = 500),
        RandomForestClassifier(n_estimators = 1000),
        RandomForestClassifier(n_estimators = 5000),
        GradientBoostingClassifier(n_estimators = 100),
        GradientBoostingClassifier(n_estimators = 500),
        GradientBoostingClassifier(n_estimators = 1000),
        GradientBoostingClassifier(n_estimators = 5000),
        MLPClassifier(hidden_layer_sizes = [5]*1) ,
        MLPClassifier(hidden_layer_sizes = [5]*2) ,
        MLPClassifier(hidden_layer_sizes = [5]*3) ,
        MLPClassifier(hidden_layer_sizes = [5]*4) ,
        MLPClassifier(hidden_layer_sizes = [5]*5) ,
        MLPClassifier(hidden_layer_sizes = [5]*10) ,
        MLPClassifier(hidden_layer_sizes = [10]*1) ,
        MLPClassifier(hidden_layer_sizes = [10]*2) ,
        MLPClassifier(hidden_layer_sizes = [10]*3) ,
        MLPClassifier(hidden_layer_sizes = [10]*4) ,
        MLPClassifier(hidden_layer_sizes = [10]*5) ,
        MLPClassifier(hidden_layer_sizes = [10]*10) ,
        MLPClassifier(hidden_layer_sizes = [20]*1) ,
        MLPClassifier(hidden_layer_sizes = [20]*2) ,
        MLPClassifier(hidden_layer_sizes = [20]*3) ,
        MLPClassifier(hidden_layer_sizes = [20]*4) ,
        MLPClassifier(hidden_layer_sizes = [20]*5) ,
        MLPClassifier(hidden_layer_sizes = [20]*10) ,
        MLPClassifier(hidden_layer_sizes = [40]*1) ,
        MLPClassifier(hidden_layer_sizes = [40]*2) ,
        MLPClassifier(hidden_layer_sizes = [40]*3) ,
        MLPClassifier(hidden_layer_sizes = [40]*4) ,
        MLPClassifier(hidden_layer_sizes = [40]*5) ,
        MLPClassifier(hidden_layer_sizes = [40]*10)]


algo = algos[algo_number]
algo_type = str(algo).split("(")[0]

print("Algo:", algo)
print("Fitting")
algo.fit(XX, yy)

pickle.dump(obj={"algo": algo},
   file=open("policy_function_traditional_{}.pkl".format(name), "wb"))

#%%
print("Evaluating MDP perfomance")
solver = KidneySolver(max_cycle_length = max_cycle_length,
              burn_in = 0)  

greedy_perf = []
this_perf = []


while True:

    env = SaidmanKidneyExchange(entry_rate  = 5,
                                death_rate  = .1,
                                time_length = 3)

    greedy = solver.greedy(env)["obj"]
    
    
    for thres in thresholds:
        
        for t in env.removed_container:
            env.removed_container[t].clear()
            
        episode_perf = []
        
        for t in range(env.time_length):
        
            idx = np.array(env.get_living(t))
            if len(idx) == 0:
                continue
            X = env.X(t)
            G, N = get_additional_regressors(env, t)
            XX = np.hstack([X, G, N])
            prob = algo.predict_proba(XX)[:,1]
            chosen = idx[prob >= thres]
            sol = solver.solve_subset(env, chosen)
            matched = list(chain(*sol["matched"].values()))
            episode_perf.append(len(matched))    
            env.removed_container[t].update(matched)
        
        this_perf.append(sum(episode_perf))
        greedy_perf.append(greedy)
        
    
    
    pickle.dump(obj={"algo": algo,
                     "greedy": greedy_perf,
                     "this": this_perf},
        file=open("policy_function_traditional_{}.pkl".format(name), "wb"))

    
    

    
    