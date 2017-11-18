#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:50:59 2017

Creates tuples of the form

(X(t), A(t), greedy_obj[t,t+h], opt_obj[t,t+h])

where [[method]]_obj[t,t+h] refers to the cardinality of 
matched pairs by [[method]] between periods t and h.

The horizons h are typically .9, .95, .99 quantiles 
of the death rate distribution.


@author: vitorhadad
"""

import numpy as np

from tqdm import trange
import pickle
from time import time
from scipy.stats import geom
import networkx as nx
import pandas as pd

from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.utils.data_utils import clock_seed, get_dead
from matching.utils.data_utils import flatten_matched, get_additional_regressors
from matching.utils.env_utils import get_actions
from matching.solver.kidney_solver2 import  optimal





if __name__ == "__main__":
    
    entry_rate = 5
    death_rate = 0.1
    time_length = 300
    burnin = 100
    
    horizons = geom(death_rate).ppf([.5, .9, .95]).astype(int)
    
    seed = clock_seed() 
    
    num_iter = 100
    num_pts_per_iter = 20
    
    data = []
    
    for i_iter in range(num_iter):
    
        env = SaidmanKidneyExchange(entry_rate  = entry_rate,
                                    death_rate  = death_rate,
                                    time_length = time_length,
                                    seed = seed)
    
        opt = optimal(env)["matched"]
        
        time_pts = np.random.randint(burnin, 
                                     time_length - max(horizons),
                                     size = num_pts_per_iter)
        
        for t in sorted(time_pts):
             
            matched_before_t = flatten_matched(opt, 0, t)
            env.removed_container[t].update(matched_before_t)
            living = env.get_living(t)
            actions = get_actions(env, t)
            
            for h in horizons:
                matched  = flatten_matched(opt, 0, t+h)
                num_dead = get_dead(env, matched, t, t+h)
                
                
            
            
    
            data.append({"X": env.X(t),
                         "A": env.A(t, "sparse"),
                         "G": G,
                         "N": N,
                         "t": t,
                         "y": y,
                         "horizon": h,
                         "opt_obj": get_rewards(opt, t, h),
                         "greedy_obj": get_rewards(greedy, t, h),
                         "max_cycle_length": max_cycle_length,
                         "entry_rate": entry_rate,
                         "death_rate": death_rate,
                         "time_length": time_length,
                         "seed": seed})
