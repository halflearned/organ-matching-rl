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

class UCB1:
    
    def __init__(self, env, t, c = 2, iters_per_arm=100, thres = 0.5):
        
        self.env = snapshot(env, t)
        self.t = t
        
        self.arms = two_cycles(self.env, t) 
        self.n_arms = len(self.arms)
        
        self.c = c
        self.r = np.zeros(self.n_arms) # Rewards
        self.n = np.zeros(self.n_arms) # Visits
        self.iters_per_arm = iters_per_arm
        self.thres = thres


    def __str__(self):
        return "UCB1(c={})".format(self.c)

        
    def simulate(self):        
        
        total_iters = self.iters_per_arm*self.n_arms
        
        for i in range(total_iters):

            # 1. Draw arm according to UCB score
            a = self.draw_arm()
            
            # 2. Take action, observe rewards
            cycle = self.arms[a]
            x = self.get_rewards(cycle) 
            
            # 3. Update statistics
            self.r[a] = (self.n[a]*self.r[a] + x)/(self.n[a]+1) 
            self.n[a] += 1
            

        
    def choose(self):
        print("Avg rewards:", self.r)
        if np.all(self.r <= 0.5):
            print("Skipping")
            return None
    
        most_visited = np.argwhere(self.n == np.max(self.n)).flatten()
        return self.arms[choice(most_visited)]
    
    
    
    def draw_arm(self):
        T = max(np.sum(self.n), 1)
        scores = self.r/(self.n+1) + self.c*np.sqrt(np.log(T)/(self.n+1))
        best = np.argwhere(scores == np.max(scores)).flatten()
        #import pdb; pdb.set_trace()
        return np.random.choice(best)
    
    
    
    def get_rewards(self, cycle):  
        snap = snapshot(self.env, self.t)
        h = max(snap.nodes[a]["death"] for a in cycle)    
        snap.populate(self.t+1, h+1, seed = clock_seed())
        return same_rewards(snap, 
                              t_begin = self.t,
                              t_end = h+1,
                              perturb = cycle) 
    
    



