#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:50:51 2017

@author: vitorhadad
"""

import numpy as np
import torch
from collections import OrderedDict

from matching.solver.kidney_solver2 import optimal, greedy, compare_optimal
from matching.utils.data_utils import  clock_seed, evaluate_policy, get_cycle_probabilities
from matching.utils.env_utils import snapshot, two_cycles
from matching.policy_function.policy_function_lstm import RNN
from matching.policy_function.policy_function_mlp import MLPNet
from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_rgcn import RGCNet

    
class MonteCarlo:
    
    def __init__(self,
                 env,
                 t,
                 priors,
                 prior_counts,
                 algo = "opt"):
        
        self.env = snapshot(env, t)
        self.t = t
        self.arms = two_cycles(self.env, t)
        self.n_arms = len(self.arms)
        self.successes = np.zeros(self.n_arms)
        self.solver = optimal
        self.horizons = [1, 5, 10, 20]
        self.successes = prior_counts * get_cycle_probabilities(self.env.get_living(t),
                                              self.arms,
                                              priors) + 1e-8
        
        
    
    def simulate(self, n_iters):

        for i in range(n_iters):
            for a in range(len(self.arms)): 
                horizon = choice(self.horizons)
                self.env.populate(self.t+1,
                              self.t+horizon+1,
                              seed = clock_seed())
                take, leave = compare_optimal(self.env, 
                          t_begin = self.t,
                          t_end = self.t+horizon+1,
                          perturb = self.arms[a])

                if take > leave:
                    self.successes[a] += 1
                
        print(self.successes)
        return self.arms[np.argmax(self.successes)]
    
    

    
    
        
#%%
if __name__ == "__main__":
    
    from sys import argv, platform
    from random import choice
    from tqdm import trange
    from os import listdir
    
    from matching.environment.optn_environment import OPTNKidneyExchange
    from matching.environment.abo_environment import ABOKidneyExchange
    from matching.environment.saidman_environment import SaidmanKidneyExchange
    from matching.utils.data_utils import get_n_matched, cumavg
    
    files = [f for f in listdir("results/") 
             if "abo" in f 
             and "txt" not in f
             and "data" not in f]    

    for i in range(1):
        
            
        if platform == "linux":
            file = choice(files)
            entry_rate = 3
            death_rate = .1
            max_time = 200    
            horizon = choice([1, 5, 10, 20])
            n_sims = choice([1, 5, 10, 20, 50, 100, 500, 1000, 2000])
            n_prior = choice([1, 5, 10, 20, 50])
            seed = 123456#clock_seed()
        else:
            file = "RNN_50-3-abo_73739624"
            entry_rate = 3
            death_rate = .1
            max_time = 200    
            n_sims = 100 #choice([1, 5, 10, 20, 50, 100, 500, 1000, 2000])
            n_prior = 50 #choice([1, 5, 10, 20, 50])
            seed = 123456#clock_seed()
            
    
        print("Opening file", file)
        try:
            net = torch.load("results/" + file)
        except Exception as e:
            print(str(e))
            continue
        
            
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

                    
        #%%
        np.random.seed(clock_seed())
        for t in range(env.time_length):
            probs, count = evaluate_policy(net,
                                           env,
                                           t,
                                           dtype= "numpy")

            for i in range(count):
                probs, _ = evaluate_policy(net,
                                           env,
                                           t,
                                           dtype= "numpy")
                cycles = two_cycles(env, t)
                if len(cycles) == 0:
                    break
                elif len(cycles) == 1:
                    res = cycles.pop()           
                else:
                    sim = MonteCarlo(env, t, probs, n_prior)
                    res = sim.simulate(n_sims)
                    
                env.removed_container[t].update(res)
                probs, count = evaluate_policy(net, env, t, dtype= "numpy")
            
                
            rewards[t] = len(env.removed_container[t])
    
            print(t, np.mean(rewards[:t+1]),
                  np.mean(g[:t+1]),
                  np.mean(o[:t+1]))
        
        with open("results/thompson_results.txt", "a") as f:
            print("{},{},{},{},{},{},{},{},{},{},{},{}"\
                .format(env_type,
                        file,
                        seed,
                        max_time,
                        int(env.entry_rate),
                        int(env.death_rate*100),
                        horizon,
                        n_sims,
                        n_prior,
                        np.mean(rewards),
                        np.mean(g),
                        np.mean(o)),
                        file = f)
 

    if platform == "darwin":
        plt.plot(cumavg(rewards), linewidth = 5);
        plt.plot(cumavg(g));
        plt.plot(cumavg(o))
