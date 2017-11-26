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
from itertools import cycle
from random import choice, shuffle
from copy import deepcopy
    
from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet
from matching.solver.kidney_solver2 import optimal, greedy
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.utils.data_utils import get_additional_regressors
from matching.tree_search.mcts import mcts
from matching.utils.data_utils import get_n_matched, clock_seed
from matching.utils.data_utils import disc_mean    
#%%


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
    from os import system
    
    if np.random.uniform() < .7:
        net_file = "RL_" + str(choice([74678542,81274922,
                           34171755,73907163,
                           90844154,58967136]))
    else:
        net_file = None
        
        
    
    burnin = 200
    entry_rate = 5
    death_rate = .1
    horizon = 10
    newseed = str(np.random.randint(1e8))
    train = True

    if net_file is not None:
        
        print("Using file: ", net_file)
        
        net = torch.load("results/" + net_file)
        
        net.gamma = 0.8
        
    else:
        
        print("New algo: ")
        
        net = choice([
               GCNet(12, None, dropout_prob = 0.2),
               GCNet(12, [100], dropout_prob = 0.2),
               GCNet(12, [50, 50], dropout_prob = 0.2),
               GCNet(12, [100, 100], dropout_prob = 0.2),
               GCNet(12, [100, 100, 100], dropout_prob = 0.2),
               GCNet(12, [100, 100, 100, 100], dropout_prob = 0.5),
               GCNet(12, [200, 200, 200], dropout_prob = 0.5),
               GCNet(12, [200, 200, 200, 200], dropout_prob = 0.5),
               GCNet(12, [200, 200, 200, 200, 200], dropout_prob = 0.5),
               ])
    
        net.gamma = choice([1, 0.99, 0.95, 0.9, 0.8, .5, .25, .1])
        
        print(net)
        
    if train:
        net.train()
    else:
        net.eval()
    
    net.horizon = horizon
    
    files = [f for f in listdir("results/")
                if f.startswith("optn_")]

    shuffle(files)
    
    net.opt = torch.optim.Adam(net.parameters(), lr = 0.001)

#%%   
    
    for k, f in enumerate(cycle(files)):

        print("Opening data", f)
        data  = pickle.load(open("results/" + f, "rb"))
        env = deepcopy(data["env"])
        o = get_n_matched(data["opt_matched"], 0, env.time_length)
        g = get_n_matched(data["greedy_matched"], 0, env.time_length)
        

        rewards = []
        actions = []
        t = -1
        print("Beginning")
        #%%
        while t < env.time_length:
            
            t += 1
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
            
            print("At t:",t, "matched ", r)
            
            msg = " File" + f.split("_")[1] + \
                  " Time:" + str(t) + \
                  " Reward" + str(r) + \
                  " Total:" + str(np.sum(rewards)) + \
                  " G:" +  str(g[:t].sum()) + \
                  " O:" +  str(o[:t].sum())
            
            print(msg)
            print(msg, file = open("RL_" + newseed + ".txt", "a"))
            
            
               
            if train and t % horizon == 0 and t >= (burnin + 2*horizon):
                
                net.opt.zero_grad()
                
                train_times = list(range(t-2*horizon, t-horizon))
                
                env_opt = deepcopy(env)
                
                for s in train_times:
                 
                    env_opt.removed_container[s].clear()
                
            
                
                for s in train_times:
                    
                    opt_m = optimal(env_opt, t_begin=s, t_end = s+2*horizon)
                    
                    baseline = disc_mean(get_n_matched(opt_m["matched"], s, s+2*horizon))
                    
                    rs = disc_mean(rewards[s:s+horizon], net.gamma)
                    
                    adv = rs - baseline
                    
                    actions[s].reinforce(torch.FloatTensor(actions[s].size()).fill_(adv))
                    
                    env_opt.removed_container[s].update(env.removed_container[s])
                    
                    
                autograd.backward([actions[s] for s in train_times])
                
                net.opt.step()
                
                
            if t % 250 == 0 and t > 500:
                
                torch.save(net, "results/RL_" + newseed)
                
                
            if platform == "linux" and t > 500 and np.sum(rewards) < (g[:t].sum()*.8):
                system("qsub job_policy.pbs")
                exit()
        
        
        
        print(newseed, k, np.sum(rewards), data["greedy_obj"], data["opt_obj"],
              file = open("results/policy_rl_results.txt", "a"))

        
        

                    
    
