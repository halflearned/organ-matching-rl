#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:49:48 2018

@author: halflearned
"""

from random import choice
from time import time
from os import system

from matching.environment.abo_environment import ABOKidneyExchange
from matching.trimble_solver.interface import optimal, greedy

entry_rate = choice([3, 5, 7])
death_rate = choice([0.05, 0.075, .1, .25, .5])
max_chain = choice([0, 1, 2])
max_cycle = choice([2, 3])
if max_chain > 0:
    frac_ndd = choice([0.05, 0.1])
else:
    frac_ndd = 0
time_length = 1000

t = time()

print("Entry:", entry_rate,
      "Death", death_rate,
      "Cycle", max_cycle,
      "Chain", max_chain)

env_params = dict(entry_rate=entry_rate,
                  death_rate=death_rate,
                  time_length=time_length,
                  fraction_ndd=frac_ndd)

envname, env = choice([  # ("OPTN", OPTNKidneyExchange(entry_rate,  death_rate, time_length)),
    # ("RSU", SaidmanKidneyExchange(entry_rate, death_rate, time_length)),
    ("ABO", ABOKidneyExchange(**env_params))])

print("\tSolving optimal")
opt = optimal(env, max_cycle=max_cycle, max_chain=max_chain)

print("\tSolving greedy")
gre = greedy(env, max_cycle=max_cycle, max_chain=max_chain)

t_diff = t - time()
res = [envname,
       env.entry_rate, env.death_rate, env.time_length, frac_ndd,
       max_cycle, max_chain,
       opt["obj"], gre["obj"],
       gre["obj"] / opt["obj"],
       t_diff]

with open("results/greedy_opt_comparison_results_trimble.txt", "a") as f:
    f.write(",".join([str(s) for s in res]) + "\n")


system("qsub job_comparison.pbs")

