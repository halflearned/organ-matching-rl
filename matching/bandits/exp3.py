#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:50:51 2017

@author: vitorhadad
"""

import numpy as np
from matching.solver.kidney_solver2 import same_rewards
from matching.utils.data_utils import  clock_seed
from matching.utils.env_utils import snapshot, two_cycles
from random import choice

#%%

class EXP3:
    
    def __init__(self, env, t, gamma=.1, iters_per_arm=100, thres = 0.5):
        
        self.env = env
        self.t = t
        self.arms = two_cycles(self.env, t) 
        self.n_arms = len(self.arms)
        self.w = np.ones(self.n_arms)
        self.p = np.full_like(self.w, fill_value = 1/self.n_arms)
        self.gamma = gamma
        self.r = np.zeros(self.n_arms)
        self.n = np.zeros(self.n_arms)
        self.iters_per_arm = iters_per_arm
        self.thres = thres
        
        
    def __str__(self):
        return "EXP3(gamma={})".format(self.gamma)
        

    def simulate(self):        
        
        total_iters = self.iters_per_arm*self.n_arms
        
        for i in range(total_iters):
            
            # 1. Update probabilities
            self.p = (1-self.gamma)*self.w/np.sum(self.w) + self.gamma/self.n_arms

            # 2. Draw action
            a = np.random.choice(self.n_arms, p = self.p)
    
            # 3. Take action, observe rewards
            cycle = self.arms[a]
            x = self.get_rewards(cycle) 
            
            # 4. Inverse propensity weight
            xhat = x/self.p[a]
            
            # 5. Update weights for chosen action
            self.w[a] *= np.exp(self.gamma*xhat/self.n_arms)
                
            # 6. Update statistic about rewards
            self.r[a] = (self.n[a]*self.r[a] + x)/(self.n[a]+1) 
            self.n[a] += 1
    
    
        
    def choose(self):
        print("Avg rewards:", self.r)
        if np.all(self.r <= self.thres):
            print("Skipping")
            return None
    
        best = np.argwhere(self.p == np.max(self.p)).flatten()
        return self.arms[choice(best)]
    
    
    def get_rewards(self, cycle):  
        snap = snapshot(self.env, self.t)
        try:
            h = max(snap.data.loc[cycle, "death"])    
        except AttributeError:
            h = max(snap.nodes[a]["death"] for a in cycle)    
        snap.populate(self.t+1, h+1, seed = clock_seed())
        return same_rewards(snap, 
                              t_begin = self.t,
                              t_end = h+1,
                              perturb = cycle)
        
        
        
        
        
        

    

        

