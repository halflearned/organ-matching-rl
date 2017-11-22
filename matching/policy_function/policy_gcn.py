#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:24:27 2017

@author: vitorhadad
"""



import numpy as np
from itertools import chain
import torch
from os import listdir
from torch import autograd
import pickle
from random import choice

from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet
from matching.solver.kidney_solver2 import optimal, greedy
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.utils.data_utils import get_additional_regressors
from matching.tree_search.mcts import mcts
from matching.utils.data_utils import get_n_matched
    

def evaluate_policy(net, env, t):
    if "GCN" in str(type(net)):
        X = env.X(t)[np.newaxis,:]
        A = env.A(t)[np.newaxis,:]
        yhat = net.forward(A, X)
        
    elif "MLP" in str(type(net)):  
        X = env.X(t)
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])
        yhat = net.forward(Z)
        
    else:
        raise ValueError("Unknown algorithm")
        
    return yhat



if __name__ == "__main__":
    

    from sys import platform    
    net_file = None
    burnin = 150
    entry_rate = 5
    death_rate = .1
    time_length = 2000
    horizon = 50
    newseed = str(np.random.randint(1e8))
    train = True
    disc = 0.97

    if net_file is not None:
        print("Using file: ", net_file)
        net = torch.load("results/" + net_file)
    else:
        print("New algo: ")
        net = choice([GCNet(12, [100], dropout_prob = 0.2),
               GCNet(12, None, dropout_prob = 0.2),
               GCNet(12, [50, 50], dropout_prob = 0.2),
               GCNet(12, [20, 20, 20], dropout_prob = 0.2),
               GCNet(12, [100, 100], dropout_prob = 0.2),
               MLPNet(26, None, dropout_prob = 0.2),
               MLPNet(26, [100], dropout_prob = 0.2),
               MLPNet(26, [50, 50], dropout_prob = 0.2),
               MLPNet(26, [100, 100], dropout_prob = 0.2),
               MLPNet(26, [20, 20, 20], dropout_prob = 0.2)])
        
        print(net)
        
    if train:
        net.train()
    else:
        net.eval()
    
    files = [f for f in listdir("results/")
                if f.startswith("optn_")]
    
    
    net.opt = torch.optim.Adam(net.parameters(), lr = 0.001)
    
        loss = n
#%%
        
    for k, f in enumerate(files):

        data  = pickle.load(open("results/" + f, "rb"))
        env = data["env"]
        env.removed_container = data["opt_matched"]
        o = get_n_matched(data["opt_matched"], env.time_length)
        g = get_n_matched(data["greedy_matched"], env.time_length)
        
        rewards = []
        actions = []
        t = 0
        #%%
        while t < env.time_length:
            
            living = np.array(env.get_living(t))
            
            if len(living) > 1:
                y_probs = evaluate_policy(net, env, t) 
            else:
                continue
            
            y_true = torch.FloatTensor(
                        np.isin(env.get_living(t), 
                                data["opt_matched"][t])\
                                .astype(float)\
                                .reshape(-1,1))
            
        
            
            
        
        
        print(newseed, k, np.sum(rewards), data["greedy_obj"], data["opt_obj"],
              file = open("results/policy_rl_results.txt", "a"))

        torch.save(net, "results/RL_" + newseed)

        if platform == "darwin":
            break
        
        


