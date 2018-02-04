#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:49:48 2018

@author: halflearned
"""

from random import choice
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.environment.abo_environment import ABOKidneyExchange
from matching.solver.kidney_solver2 import optimal, greedy
from time import time

while True:

    t = time()
    entry_rate = choice([1,2,3,4,5,6,7,8,9,10])
    death_rate = choice([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08 ,0.09, .1,.2,.3,.4,.5,.6,.7,.8,.9])
    mcl = choice([2])
    time_length = 1000
    (envname, env) = choice([("OPTN", OPTNKidneyExchange(entry_rate,  death_rate, time_length)), 
                             ("RSU",  SaidmanKidneyExchange(entry_rate, death_rate, time_length)),
                             ("ABO",  ABOKidneyExchange(entry_rate, death_rate, time_length))])
                        
    print(str(env))

    opt = optimal(env, max_cycle_length = mcl)
    gre = greedy(env, max_cycle_length = mcl)
    t_diff = t-time()

    res = [envname, env.entry_rate, env.death_rate, env.time_length, mcl, opt["obj"], gre["obj"], gre["obj"]/opt["obj"], t_diff]

    with open("results/greedy_opt_comparison_results.txt", "a") as f:
        f.write(",".join([str(s) for s in res]) + "\n")
