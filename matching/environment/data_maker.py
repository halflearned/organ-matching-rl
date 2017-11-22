#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:50:33 2017

@author: vitorhadad
"""

from sys import argv
import numpy as np
import pickle

from matching.environment.abo_environment import ABOKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.solver.kidney_solver2 import optimal, greedy

if len(argv) == 1:
    env_name = "optn"
    entry_rate = 5
    death_rate = 0.1
    time_length = 2000
else:
    env_name = argv[1]
    entry_rate = int(argv[2])
    death_rate = float(argv[3])
    time_length = int(argv[4])
    
    
seed = np.random.randint(1e8)
 
envclass = {"optn": OPTNKidneyExchange,
            "saidman": SaidmanKidneyExchange,
            "abo": ABOKidneyExchange}
   
env = envclass[env_name](entry_rate, 
                         death_rate,
                         time_length,
                         seed = seed)

o = optimal(env)
g = greedy(env)
#%%
pickle.dump(obj= {"env": env,
                  "entry_rate": entry_rate,
                  "death_rate": death_rate,
                 "opt_matched": o["matched"],
                 "greedy_matched": g["matched"],
                 "opt_obj": o["obj"],
                 "greedy_obj": g["obj"],
                 "seed": seed},
            file= open("results/" + "_".join((env_name,str(seed),".pkl")), "wb"))




