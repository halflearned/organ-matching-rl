#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:50:51 2017

@author: vitorhadad
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence #, pad_packed_sequence

from matching.policy_function.policy_function_rnn import RNN
from matching.solver.kidney_solver2 import optimal, greedy
from matching.utils.data_utils import get_size, clock_seed, cumavg, get_additional_regressors
from matching.utils.env_utils import snapshot, two_cycles





class OptimalSimulationWithPriors:
    
    def __init__(self, env, t, 
                 n_iters = 10, 
                 method = "cycles",
                 mix = 0.5,
                 none_prob = None):
        
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
        self.rnn = torch.load("results/policy_function_lstm")
        self.mix = mix
        self.none_prob = none_prob
    
    
    def simulate(self, horizon, n_iters):
        arm_values = dict.fromkeys(self.arms, 0)        
        for i in range(n_iters):
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
        return arm_values
    
    
    
    def optimize(self, env, t):
        if self.method == "cycles":
           pulled_arms = optimal(env, t, t)["matched_cycles"][t]
           pulled_arms = list(map(lambda x: tuple(sorted(x)), pulled_arms))
        else:
           pulled_arms = optimal(env, t, t)["matched"][t]
        return pulled_arms
    
    
    def evaluate_rnn(self, net, env, t):
        n = len(env.get_living(t))
        lens = [n]
        X = env.X(t)
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])
        er = np.full((Z.shape[0], 1), fill_value = env.entry_rate)
        dr = np.full((Z.shape[0], 1), fill_value = env.death_rate)
        ZZ = np.hstack([Z, er, dr])[np.newaxis,:,:].transpose((1,0,2))
        ZZv = Variable(torch.FloatTensor(ZZ))
        ZZv = pack_padded_sequence(ZZv, lens)
        outputs,_ = net.forward(ZZv)  
        p = net.out_layer(outputs[:n,0])
        yhat = F.softmax(p, dim= 1)[:,1]   
        return yhat.data.numpy()
    
    
    def get_priors(self):
        liv_idx = {j:i for i,j in enumerate(self.env.get_living(self.t))}
        probs = self.evaluate_rnn(self.rnn, self.env, self.t)
        cycle_values = dict()
        if self.none_prob is not None:
            none_prob = self.none_prob
        else:
            none_prob = 1/self.n_arms
        for a in self.arms:
            if a is not None:
                cycle_values[a] = np.mean((probs[liv_idx[a[0]]],
                                           probs[liv_idx[a[1]]]))
            else:
                cycle_values[None] = none_prob
        return cycle_values
        
    
    def choose(self, horizon = 10, n_iters = 10, thres = None):
        if len(self.arms) == 1:
            return None
        rnn_probs = self.get_priors()
        sim_probs = self.simulate(horizon, n_iters)
        probs = {v:self.mix*rnn_probs[v]+(1-self.mix)*sim_probs[v] for v in rnn_probs}
        a = max(probs, key = lambda x: probs[x])
        if thres is not None:
            if probs[a] >= thres:
                return a
            else:
                return None
        else:
            return a
        
    
    
#%%
if __name__ == "__main__":
    
    from matching.environment.optn_environment import OPTNKidneyExchange
    from matching.utils.data_utils import get_n_matched
    from random import choice
    from sys import platform
    
    seed = clock_seed()
    
    
    if platform == "linux":
        entry_rate = choice([3, 5, 10])
        death_rate = choice([.01, .05, .1])
        n_iters = choice([1, 10, 20, 100])
        horizon = choice([5, 10, 20])
        mix = choice([0, .25, .5, .75, 1])
        none_prob = choice([0, None])
        thres_coeff = np.random.uniform(0, .5, size = 2)
    else:
        entry_rate = 3
        death_rate = .01
        n_iters = 5
        horizon = 10
        mix = .5
        none_prob = 0
        thres_coeff = 0.5, .01
        
        
    max_expected_size = entry_rate/death_rate
    
    env = OPTNKidneyExchange(5, .1, 500, 
                             seed = 12345,
                             initial_size = 0)
    
    opt = optimal(env)
    gre = greedy(env)
    
    o = get_n_matched(opt["matched"], 0, env.time_length)
    g = get_n_matched(gre["matched"], 0, env.time_length)
    
    rewards = np.zeros(env.time_length)

    #%%
    for t in range(env.time_length):
    
        while True:
        
            optsim = OptimalSimulationWithPriors(
                             env, t,      
                             none_prob = none_prob,
                             mix = mix)
            
            if thres_coeff is not None:
                size = len(env.get_living(t))/max_expected_size
                thres = thres_coeff[0] * size ** thres_coeff[1]
            else:
                thres = None
            a = optsim.choose(horizon, n_iters, thres)
            
            
            if a is not None:
                env.removed_container[t].update(a)
                rewards[t] += len(a)
            
            else:
                break
            
            
        print("Time:", t,
              "R:", rewards[:t].mean(),
              "G:", g[:t].mean(),
              "O:", o[:t].mean())
        
    
    print(env.entry_rate,
          env.death_rate,
          env.time_length,
          seed,
          horizon,
          n_iters,
          none_prob,
          thres_coeff[0],
          thres_coeff[1],
          rewards[:t].mean(),
          g[:t].mean(),
          o[:t].mean(),
          file = open("results/optsim_results_priors_and_threshold.txt", "a"))
    
              
              
