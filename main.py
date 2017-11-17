#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7th 17:44:35 2017

MCTS with policy function

"""


import numpy as np

from matching.solver.kidney_solver2 import  optimal, greedy
import matching.tree_search.mcts_with_opt_rollout as mcts

    
from collections import defaultdict
from random import choice
from os import listdir
import torch
from sys import platform

from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.utils.env_utils import snapshot, get_environment_name

#%%
er = 5
dr = .1
time_length = 200
    
    
for seed in range(17, 1000):
    
    if platform != "darwin":
        scl = .01
        tpa = 100
        t_horiz = 2
        r_horiz = 45
        n_rolls = 7
        net_file = None
        burnin = 0
        
    else:
        net_files = [f for f in listdir("results/") if 
                f.startswith("MLP_") or 
                f.startswith("GCN_")] + [None]
        scl = choice([0.001, 0.1, 0.5, 1])
        tpa = choice([1,5,10,20,50])
        t_horiz = choice([2, 5, 10])
        r_horiz = choice([1, 10, 22, 45])
        n_rolls = choice([1, 5])
        net_file = None
        burnin = 100

    print("USING:")
    print("scl", scl)
    print("tpa", tpa)
    print("t_horiz", t_horiz)
    print("r_horiz", r_horiz)
    print("n_rolls", n_rolls)
    print("net", net_file)


    config = (scl, tpa, n_rolls, t_horiz, r_horiz, net_file)

    if net_file is not None:
        net = torch.load("results/" + net_file)
    else:
        net= None
    
    opt = None
    g   = None
    
    name = str(seed)        

    env = OPTNKidneyExchange(entry_rate  = er,
            death_rate  = dr,
            time_length = time_length,
            seed = seed,
            populate = True)
    
    print("Beginning")

    matched = defaultdict(list)
    rewards = 0                
#%%    
    t = 0
    while t < env.time_length:
        
        print("\nStarting ", t)
        root = mcts.Node(parent = None,
                    t = t,
                    reward = 0,
                    env = snapshot(env, t),
                    taken = None,
                    actions = mcts.get_actions(env, t))

        iters = 0
    
        print("Actions: ", root.actions)
        n_act = len(root.actions)

        if n_act > 1:    
            a = choice(root.actions)
            n_iters = int(tpa * n_act)
             
            for i_iter in range(n_iters):
                
                mcts.run(root,
                    scalar = scl,
                    tree_horizon = t_horiz,
                    rollout_horizon = r_horiz,
                    net = net,
                    n_rollouts = n_rolls)
                
                
            a = mcts.choose(root)
            print("Ran for", n_iters, "iterations and chose:", a)
    
        else:
            
            a = root.actions[0]
            print("Chose the only available action:", a)

        
        if a is not None:
            
            print("Staying at t.")
            assert a[0] not in env.removed(t)
            assert a[1] not in env.removed(t)
            env.removed_container[t].update(a)
            matched[t].extend(a)
            rewards += len(a)
        
        else:
        
            print("Done with", t, ". Moving on to next period\n")
            t += 1
    
    
#%%
          
    

    env = env.__class__(entry_rate  = er,
            death_rate  = dr,
            time_length = time_length,
            seed = seed)

    opt = optimal(env)#["obj"]
    g = greedy(env)#["obj"]


#%%
    
    this_matched = mcts.flatten_matched(matched, burnin)
    g_matched = mcts.flatten_matched(g["matched"], burnin)
    opt_matched = mcts.flatten_matched(opt["matched"], burnin)
    
    n = len(env.get_living(burnin, env.time_length))
    
    g_loss = len(mcts.get_dead(env, g_matched, burnin))/n
    opt_loss = len(mcts.get_dead(env, opt_matched, burnin))/n
    this_loss = len(mcts.get_dead(env, this_matched, burnin))/n

    print("MCTS loss: ", this_loss)
    print("GREEDY loss:", g_loss)
    print("OPT loss:", opt_loss)
    
    envname = get_environment_name(env)
    
    results = [envname,seed,er,dr,time_length,*config,this_loss,g_loss,opt_loss]


    with open("results/mcts_results.txt", "a") as f:
        s = ",".join([str(s) for s in results])
        f.write(s + "\n")


    if platform == "darwin":
        break
    
    
#%%
import matplotlib.pyplot as plt

gc = get_n_matched(g["matched"]).cumsum()
tc = get_n_matched(matched).cumsum()
oc = get_n_matched(opt["matched"]).cumsum()
ts = np.arange(1, len(gc)+1)

plt.plot(ts, gc/ts, label = "greedy", color = "orange")
plt.plot(ts, oc/ts, label = "opt", color = "blue")
plt.plot(ts, tc/ts, label = "this", color = "green")








