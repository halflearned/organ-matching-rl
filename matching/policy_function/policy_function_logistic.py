#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:24:27 2017

@author: vitorhadad
"""



import numpy as np
from itertools import chain
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from os import listdir
from torch import autograd
import pickle
from random import choice, shuffle
from copy import deepcopy
from itertools import cycle
from torch.nn import SELU, ReLU
    
from matching.policy_function.policy_function_agcn import AGCNet
from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet
from matching.solver.kidney_solver2 import optimal, greedy
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange

from matching.utils.data_utils import get_additional_regressors
from matching.tree_search.mcts import mcts
from matching.utils.data_utils import get_n_matched, clock_seed
from matching.utils.data_utils import disc_mean, cumavg

#%%

def evaluate_policy(net, env, t):
    
    if "AGCN" in str(type(net)):
        X = env.X(t)
        G,N = get_additional_regressors(env, t)
        A = env.A(t)
        yhat = net.forward(A, X, G, N)
        
    elif "GCN" in str(type(net)):
        X = env.X(t)[np.newaxis,:]
        A = env.A(t)[np.newaxis,:]
        yhat = net.forward(A, X)
        
    elif "MLP" in str(type(net)):  
        X = env.X(t)
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])
        etc1 = np.full_like(Z[:,:1], fill_value = env.entry_rate)
        etc2 = np.full_like(Z[:,:1], fill_value = env.death_rate)
        ZZ = np.c_[ Z, etc1, etc2]
        ZZ = np.hstack([ZZ, ZZ**2])
        yhat = net.forward(ZZ)
        
    else:
        raise ValueError("Unknown algorithm")
        
    return yhat


#%%


if __name__ == "__main__":
    

    from sys import platform
    from os import system
   
    
    time_length = 201
    burnin = 50
    entry_rate = 5
    death_rate = .1
    
    horizon = 22 
    newseed = str(np.random.randint(1e8))
    train = True
    
    # Fill weights from output of logistic regression
    print("New algo: ")
    logistic = pickle.load(open("results/optn_logistic2.pkl", "rb"))
    n_regressors = max(logistic.coef_.shape)
    net = MLPNet(n_regressors, hidden_sizes = None, dropout_prob = 0)
    list(net.parameters())[0].data = torch.FloatTensor(logistic.coef_.reshape(1,-1))
    list(net.parameters())[1].data = torch.FloatTensor(logistic.intercept_)
    
    
    gamma = 0.99
    
    print(net)
    
    net.opt = torch.optim.Adam(net.parameters(), lr = 0.0001)    

    
#%%   
    
    for epoch in range(5):
        
        for seed in [0,1,2]*5:
        
            
            print("Creating environment")
            env = OPTNKidneyExchange(entry_rate, 
                                     death_rate,
                                     time_length,
                                     seed = seed)
            
            print("Solving environment")
            opt = optimal(env)
            gre = greedy(env)
            
            o = get_n_matched(opt["matched"], 0, env.time_length)
            g = get_n_matched(gre["matched"], 0, env.time_length)
            
            rewards = []
            actions = []
            t = 0
            print("Beginning")
            
            for t in range((2*horizon+1)*(epoch+1)):
                
                print("Getting living")
                living = np.array(env.get_living(t))
                if len(living) > 1:
                    a_probs = evaluate_policy(net, env, t) 
                else:
                    print("No one was alive. Continuing.")
                    continue
                
                print("Size {}".format(len(living)), end = "\t")
                print("E[Prob] {:1.2f}".format(a_probs.mean().data[0]), end = "\t")
                print("Std[Prob] {:1.2f}".format(a_probs.std().data[0]))
    
                print("Selecting")
                a = a_probs.squeeze(0).bernoulli()
                selected = a.data.numpy().flatten().astype(bool)
                
                s = optimal(env, 
                          t_begin = t,
                          t_end = t,
                          subset = living[selected])
                   
                
                mat = s["matched"][t]
                r   = s["obj"]
                
                env.removed_container[t].update(mat)
                
                actions.append(a)
                rewards.append(r)
                
                #print("At t:",t, "matched ", r)
                
                msg = " Time:" + str(t) + \
                      " Reward" + str(r) + \
                      " Total:" + str(np.sum(rewards)) + \
                      " G:" +  str(g[:t].sum()) + \
                      " O:" +  str(o[:t].sum())
                
                print(msg)
                print(msg, file = open("RL_" + newseed + ".txt", "a"))
                
                
                   
                if train and t % horizon == 0 and t >= (burnin + 2*horizon):
                    
                    print("TRAINING")
                    
                    net.opt.zero_grad()
                    
                    train_times = list(range(t-2*horizon, t-horizon))
                    
                    env_opt = deepcopy(env)
                    
                    for s in train_times:
                     
                        env_opt.removed_container[s].clear()
                    
                
                    
                    for s in train_times:
                        
                        opt_m = optimal(env_opt, t_begin=s, t_end = s+2*horizon)
                        
                        baseline = disc_mean(get_n_matched(opt_m["matched"], s, s+2*horizon),
                                                           gamma)
                        
                        rs = disc_mean(rewards[s:s+horizon], gamma)
                        
                        adv = rs - baseline
                        
                        actions[s].reinforce(torch.FloatTensor(actions[s].size()).fill_(adv))
                        
                        env_opt.removed_container[s].update(env.removed_container[s])
                        
                        
                    autograd.backward([actions[s] for s in train_times])
                    
                    net.opt.step()
                    
                if t % 500 == 0 and t > 500:
                    
                    torch.save(net, "results/RL_" + newseed)
                    
                    
                if platform == "linux" and t > 500 and np.sum(rewards) < (g[:t].sum()*.8):
                    system("qsub job_policy.pbs")
                    exit()
    
            
            
            if platform != "linux":
    
                fig = plt.figure()
                plt.plot(cumavg(rewards[:t]), linewidth=4);
                plt.plot(cumavg(o[:t]));
                plt.plot(cumavg(g[:t]));
                plt.title("GCN:{}, $\gamma$:{}"\
                          .format(net.hidden_sizes, gamma))
                fig.savefig("results/RL_" + newseed)
                
        