#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:50:51 2017

@author: vitorhadad
"""

import numpy as np


from matching.solver.kidney_solver2 import optimal_with_discount, optimal, greedy
from matching.utils.data_utils import  clock_seed
from matching.utils.env_utils import snapshot, two_cycles




class OptimalSimulation:
    
    def __init__(self, env, t, n_iters = 10, method = "cycles"):
        self.env = snapshot(env, t)
        self.t = t
        self.n_iters = n_iters
        self.method = method
        
        if method == "cycles": 
            self.arms = list(map(lambda x: tuple(sorted(x)), 
                                 two_cycles(self.env, t))) + [None]
        elif method == "pairs":
            self.arms = self.env.get_living(self.t) + [None]
        else:
            raise ValueError("Unknown optimal simulation method.")
        
        self.n_arms = len(self.arms)
        
    
    def simulate_optimal(self, horizon=10, n_iters = 10):
        arm_values = dict.fromkeys(self.arms, 0)        
        for i in range(n_iters):
            print(i)
            self.env.populate(self.t+1,
                              self.t+horizon+1,
                              seed = clock_seed())  
            pulled_arms = self.optimize(self.env, self.t)
            if len(pulled_arms) > 0:
                for a in pulled_arms:
                    arm_values[a] += 1
            else:
                arm_values[None] += 1
                
        for k,v in arm_values.items():
            arm_values[k] = v/n_iters
        
        print(arm_values.values())
        return arm_values
    
    
    
    def optimize(self, env, t):
        if self.method == "cycles":
           pulled_arms = optimal(env, t, t)["matched_cycles"][t]
           pulled_arms = list(map(lambda x: tuple(sorted(x)), pulled_arms))
        else:
           pulled_arms = optimal(env, t, t)["matched"][t]
        return pulled_arms
    
    
    
        
if __name__ == "__main__":
    
    from sys import argv
    from random import choice
    
    #from matching.environment.optn_environment import OPTNKidneyExchange
    from matching.environment.abo_environment import ABOKidneyExchange
    #from matching.environment.saidman_environment import SaidmanKidneyExchange
    from matching.utils.data_utils import get_n_matched, cumavg
    
    if len(argv) > 1:
        entry_rate = int(argv[1])
        death_rate = float(argv[2])
        horizon = int(argv[3])
        n_iters = int(argv[4])
        max_time = int(argv[5])
        seed = int(argv[6])
    else:
        entry_rate = 3
        death_rate = .1
        horizon = 5
        n_iters = 10
        max_time = 500
        seed = 12345
        
       
    env_type = "abo"
    env = ABOKidneyExchange(entry_rate,
                            death_rate, 
                            max_time, 
                            seed = seed)
    
    opt = optimal(env)
    gre = greedy(env)
    
    o = get_n_matched(opt["matched"], 0, env.time_length)
    g = get_n_matched(gre["matched"], 0, env.time_length)
    
    rewards = np.zeros(env.time_length)

    
    outfile = "{}_optsim_entry{}_death{}_hor{}_nit{}_seed{}"\
            .format(env_type,
                    int(env.entry_rate),
                    int(env.death_rate*100),
                    horizon,
                    n_iters,
                    seed)
    

    s = clock_seed()
    print("Updated seed:", s)
    print("Horizon", horizon)
    print("N_iters", n_iters)
    np.random.seed(s)
    for t in range(env.time_length):
    
        while True:
        
            optsim = OptimalSimulation(env, t)
            if optsim.n_arms == 1:
                a = None
            else:
                probs = optsim.simulate_optimal(horizon=horizon, n_iters=n_iters)
                a = max(probs, key = lambda x: probs[x])
            
            if a is not None:
                env.removed_container[t].update(a)
                rewards[t] += len(a)
            
            else:
                break
            
        with open("results/optsim_data/".format(env_type) + outfile + ".txt", "a") as f:
            print(rewards[t], g[t], o[t], file = f)
            
    with open("results/optsim_results.txt", "a") as f:
        print(env_type, int(env.entry_rate), int(100*env.death_rate), seed,env.time_length,
               horizon, n_iters, np.sum(rewards), np.sum(g), np.sum(o), file = f) 
       
    print(np.sum(rewards), np.sum(g), np.sum(o))    
