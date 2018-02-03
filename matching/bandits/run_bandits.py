#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:37:48 2018

@author: vitorhadad
"""

#%%

from sys import platform
from random import choice
import numpy as np

from matching.utils.env_utils import two_cycles
from matching.solver.kidney_solver2 import optimal, greedy
from matching.utils.data_utils import  clock_seed, get_n_matched

from matching.environment.abo_environment import ABOKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange

from matching.bandits.exp3 import EXP3
from matching.bandits.ucb1 import UCB1
from matching.bandits.thompson import Thompson

n_iters = 1 if platform == "darwin" else 10000
#%%
for i in range(n_iters):
    
    if platform == "linux":
        env_type = "abo"
        entry_rate = choice([5, 7, 10])
        death_rate = choice([.1, .08, .05]) 
        max_time = 1001
        thres = choice([.5, .7, .9])
        seed = clock_seed()
        algorithm = choice(["EXP3",
                            "UCB1",
                            "Thompson"])  
        env = choice([ABOKidneyExchange,
                      SaidmanKidneyExchange,
                      OPTNKidneyExchange])
        gamma = choice([.01, .05, .1, .5])
        c = choice([.01, 0.05, .1, .5])
    else:
        env_type = "abo"
        entry_rate = choice([3]) 
        death_rate = choice([.1])          
        max_time = 10
        seed = 126296
        thres = choice([.5])
        algorithm = "Thompson"
        env = OPTNKidneyExchange
        gamma = .1
        c = 2

                      
                      
    env = env(entry_rate, death_rate, max_time)
        
    opt = optimal(env)
    gre = greedy(env)
    o = get_n_matched(opt["matched"], 0, env.time_length)
    g = get_n_matched(gre["matched"], 0, env.time_length)

    rewards = np.zeros(env.time_length)
    log_every = 1

    np.random.seed(seed)
    #%%

    for t in range(env.time_length):
        while True:
            cycles = two_cycles(env, t)
            if len(cycles) == 0:
                break
            else: 
                if algorithm == "EXP3":
                    algo = EXP3(env, t, gamma=gamma, thres=thres)
                elif algorithm == "Thompson":
                    algo = Thompson(env, t, thres=thres)
                elif algorithm == "UCB1":
                    algo = UCB1(env, t, c=c, thres=thres)
                    
                algo.simulate()
                res = algo.choose()
                if res is not None:
                    env.removed_container[t].update(res)
                    rewards[t] = len(env.removed_container[t])
                else:
                    break
        
        if t == env.time_length - 1:
            rewards[t] += len(optimal(env, t, t)["matched_pairs"])
        
        if algorithm == "EXP3":
            param = gamma
        elif algorithm == "Thompson":
            param = np.nan
        elif algorithm == "UCB1":
            param = c
        
        
        if t % log_every == 0 and t > 0:
            stats=[algorithm,
                   param,
                   thres,
                   "\"" + str(env) + "\"",
                   seed,
                   t,
                   int(env.entry_rate),
                   int(env.death_rate*100),
                   rewards[t],
                   g[t],
                   o[t]]
            msg = ",".join(["{}"]*len(stats)).format(*stats)
        
            if platform == "linux":
                with open("results/bandit_results3.txt".format(env_type), "a") as f:
                    print(msg, file = f)
            else:    
                print(t, np.sum(rewards[:t+1]),
                          np.sum(g[:t+1]),
                          np.sum(o[:t+1]))
           
            