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

from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet
from matching.solver.kidney_solver2 import optimal, greedy
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.utils.data_utils import get_additional_regressors
from matching.tree_search.mcts import mcts


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



#%%
    
if __name__ == "__main__":
    

    from os import listdir
    from random import choice
    from pandas import Series
    from sys import platform
    
    from matching.utils.data_utils import get_n_matched
    
    #net_file = choice([f for f in listdir("results/") if
                       #f.startswith("GCN") or 
    #                   f.startswith("MLP")])
    
    net_file = "GCN_50_10_217802.pkl"

    burnin = 20
    entry_rate = 5
    death_rate = .1
    time_length = 2000
    horizon = 200
    newseed = str(np.random.randint(1e8))
    train = True
    disc = 0.97

    if net_file is not None:
        print("Using file: ", net_file)
        net = torch.load("results/" + net_file)
    else:
        print("New algo: ")
        net = MLPNet(24, [100], dropout_prob = 0.1)#GCNet(10, [100], dropout_prob = 0.1)
        print(net)
        
    if train:
        net.train()
    else:
        net.eval()
    
    files = [f for f in listdir("results/")
                if f.startswith("optn_")]
    
    

#%%
        
    for f in files:

        data  = pickle.load(open("results/" + f, "rb"))
        env = data["env"]
        env.removed_container = data["opt_matched"]
        o = get_n_matched(data["opt_matched"], env.time_length)
        g = get_n_matched(data["greedy_matched"], env.time_length)
        

        rewards = []
        actions = []
        
        for t in range(100): #range(env.time_length):
            
            living = np.array(env.get_living(t))
            if len(living) > 0:
                a_probs = evaluate_policy(net, env, t) 
            else:
                continue
            
            a = a_probs.squeeze(0).bernoulli()
            selected = a.data.numpy().flatten().astype(bool)
            
            s = optimal(env, 
                      t_begin = t,
                      t_end = t,
                      restrict = living[selected])
               
            
            mat = s["matched"][t]
            r   = s["obj"]
            
            env.removed_container[t].update(mat)
            
            if t % 10 == 0: print(t)
            
            actions.append(a)
            rewards.append(r)
            
                                  
        # Episode is over
        if train:
            net.opt.zero_grad()
            
            train_times = list(range(burnin, env.time_length - horizon))
            
            for s in train_times:
                baseline = np.sum([disc**i * r for i,r in enumerate(o[s:s+horizon])])
                rs = np.sum([disc**i * r for i,r in enumerate(rewards[s:s+horizon])])
                delta = rs - baseline
                actions[s].reinforce(torch.FloatTensor(actions[s].size()).fill_(delta))
                
            autograd.backward([actions[s] for s in train_times])
            net.opt.step()
            
        
        print("Matched (this): \t", np.sum(rewards))
        print("Matched (greedy): \t", g["obj"])
        print("Matched (opt): \t", o["obj"])
        
        
        torch.save(net, "results/RL_" + newseed)
