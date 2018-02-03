#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:51:11 2018

@author: vitorhadad
"""

import numpy as np
from matching.solver.kidney_solver2 import same_rewards
from matching.utils.data_utils import  clock_seed
from matching.utils.env_utils import snapshot, two_cycles
from random import choice

class Thompson:
    
    def __init__(self, env, t,
                 alphas = None,
                 betas = None,
                 thres = 0.5,
                 iters_per_arm=100):
        
        self.env = snapshot(env, t)
        self.t = t
        self.arms = two_cycles(self.env, t) 
        self.n_arms = len(self.arms)
        self.iters_per_arm = iters_per_arm
        self.thres = thres

        
        # Prior successes
        if alphas is None:
            self.alphas = np.ones(self.n_arms)
        else:
            self.alphas = alphas
        
        # Prior failures
        if betas is None:
            self.betas = np.ones(self.n_arms)
        else:
            self.betas = alphas            
        
        self.s = np.zeros(self.n_arms) # Successes
        self.f = np.zeros(self.n_arms) # Failures
        
        self.r = np.zeros(self.n_arms) # Rewards
        self.n = np.zeros(self.n_arms) # Visits


    def __str__(self):
        return "Thompson"

        
    def simulate(self):        
        
        total_iters = self.iters_per_arm*self.n_arms
        
        for i in range(total_iters):

            # 1. Draw arm according to UCB score
            a = self.draw_arm()
            
            # 2. Take action, observe rewards
            cycle = self.arms[a]
            x = self.get_rewards(cycle) 
            
            # 3. Update data
            self.s[a] += x
            self.f[a] += 1-x
            
            # 4. Update parameters
            self.r[a] = (self.n[a]*self.r[a] + x)/(self.n[a]+1) 
            self.n[a] += 1
            


    def choose(self):
        print("Avg rewards:", self.r)
        if np.all(self.r <= self.thres):
            print("Skipping")
            return None
    
        alpha_post = self.alphas + self.s
        best = choice(np.argwhere(alpha_post == np.max(alpha_post)).flatten())
        return self.arms[best]
    
    
    
    def draw_arm(self):
        alpha_post = self.alphas + self.s
        beta_post = self.betas + self.f
        thetas = np.random.beta(alpha_post, beta_post)
        return np.argmax(thetas)
    
    
    
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
        
        
        
        
        
        