#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:50:56 2018

@author: vitorhadad
"""
#
import numpy as np
import networkx as nx
import os

from tqdm import trange
from matching.solver.kidney_solver2 import optimal, greedy, get_two_cycles
from matching.utils.data_utils import clock_seed, evaluate_policy, get_n_matched, get_cycle_probabilities
from matching.utils.env_utils import snapshot, two_cycles
from matching.policy_function.policy_function_lstm import RNN
from matching.policy_function.policy_function_mlp import MLPNet
from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_rgcn import RGCNet
from matching.policy_function.policy_function_attention import AttentionRNN
from matching.environment.abo_environment import ABOKidneyExchange


#%%
def run_node2vec(G, 
                 path = "",
                 input = "edges.txt",
                 output = "emb.txt",
                 d = 10):
    
    nx.write_edgelist(G, path + input, data = False)
    
    cmd = "./node2vec -i:{}{} -o:{}{} -d:{} -dr"\
            .format(path, input, path, output, d)
    
    os.system(cmd)
    
    with open("emb.txt", "r") as emb:
        lines = emb.readlines()
        n = len(lines) - 1 
        features = np.zeros(shape = (n, d)) 
        for k, line in enumerate(lines[1:]):
            _, *xs = line.split(" ")
            try:
                features[k] = [float(x) for x in xs]
            except:
                import pdb; pdb.set_trace()
    return features
        

    
def get_features(env):
    opt= optimal(env)
    
    features = []   
    labels = []
    
    for t in range(env.time_length):
        liv = np.array(env.get_living(t))
        A = env.A(t)
        has_cycle = np.diag(A @ A) > 0
        liv = liv[has_cycle]
        
        m = opt["matched"][t]
        
        Y = np.zeros(len(liv)) 
        Y[np.isin(liv, list(m))] = 1
        labels.append(Y)
        

        if len(liv) > 0:
            X = env.X(t)[has_cycle]
            subg = env.subgraph(liv)
            E = run_node2vec(subg) 
            features.append(np.hstack([X, E]))
        
        env.removed_container[t].update()
           
        
    return np.vstack(features), np.hstack(labels)
    
    
    
env = ABOKidneyExchange(entry_rate = 5,
                        death_rate = .1, 
                        time_length = 10, 
                        seed = clock_seed())

X, Y = get_features(env)
np.save("X.npy", X)
np.save("Y.npy", Y)



#%%

    


#%%
#

#


#%%
#
#env = ABOKidneyExchange(entry_rate = 5,
#                        death_rate = .1, 
#                        time_length = 1500, 
#                        seed = clock_seed())
#
#opt = optimal(env)
#gre = greedy(env)
#
##%%
#def evaluate(algo, env, thres):
#
#    env.removed_container.clear()
#    rewards = []
#    for t in trange(env.time_length):
#        
#        liv = np.array(env.get_living(t))
#        A = env.A(t)
#        has_cycle = np.diag(A @ A) > 0
#        liv_and_cycle = liv[has_cycle]
#        yhat_full = np.zeros(len(liv), dtype=bool)
#        
#        if len(liv_and_cycle) == 0:
#            continue
#        
#        X = env.X(t)[has_cycle]
#        subg = env.subgraph(liv_and_cycle)
#        E = run_node2vec(subg) 
#        F = np.hstack([X, E])
#    
#        yhat = algo.predict_proba(F)[:,1] > thres
#        yhat_full[has_cycle] = yhat
#        potential = liv[yhat_full]
#        
#        removed = optimal(env, t, t, subset=potential)["matched"][t]
#        env.removed_container[t].update(removed)
#        rewards.append(len(removed))
#        
#    return rewards
#
#
#r = evaluate(pipe, env, .05)
#
#gre_n = get_n_matched(gre["matched"], 0, env.time_length)
#opt_n = get_n_matched(opt["matched"], 0, env.time_length)

#print("\nrewards\n",
#      np.sum(r[500:]),
#      np.sum(gre_n[500:]),
#      np.sum(opt_n[500:]))
#    