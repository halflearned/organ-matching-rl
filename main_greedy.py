#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7th 17:44:35 2017

MCTS with policy function

"""


import numpy as np

    
from collections import defaultdict
from random import choice
from os import listdir, system
import torch
from sys import platform
import pickle


import matching.tree_search.mcts_greedy as mcts
from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.utils.env_utils import  get_environment_name
from matching.utils.data_utils import get_n_matched

#%%


if platform == "linux":
    scl, criterion = choice([(.1,  "visits"),
                             (.5,   "visits"),
                             (1,    "visits"),
                             (2,    "visits")])
    
    tpa = choice([1, 5])
    t_horiz = choice([1, 5])
    r_horiz = choice([5, 20, 100])
    n_rolls = choice([1, 10])
    net_file = choice(["RL_23101647",
                       "RL_26213785",
                       "RL_73162545",
                       "RL_74678542",
                       "RL_81274922",
                       None])
    gamma = choice([1, .95, .9, .75, .5, .1])
    file = 'optn_51332267_.pkl'#choice([f for f in listdir("results/")
                # if f.startswith("optn_")])
    
else: 
    scl = .1
    criterion = "visits"
    tpa = 1
    t_horiz = 1
    r_horiz = 100
    net_file = None
    n_rolls = 5
    gamma = .9 
    file = 'optn_51332267_.pkl'
    


if net_file is not None:
    net = torch.load("results/" + net_file)
    net.eval()
else:
    net = None


config = (scl, criterion, tpa, n_rolls, 
          t_horiz, r_horiz, net_file, gamma)



name = str(np.random.randint(1e8))      

prefix = "MCTSG2_"    

logfile = prefix + name + ".txt"

data  = pickle.load(open("results/" + file, "rb"))
env = data["env"]
envname = get_environment_name(env)
o = get_n_matched(data["opt_matched"], 0, env.time_length)
g = get_n_matched(data["greedy_matched"], 0, env.time_length)

matched = defaultdict(list)
rewards = np.zeros(env.time_length)


print("scl " + str(scl) + \
      " tpa " + str(tpa) + \
      " t_horiz " + str(t_horiz) + \
      " r_horiz " + str(r_horiz) + \
      " n_rolls " + str(n_rolls) + \
      " net " + str(net_file) + \
      " gamma " + str(gamma),
      file = open(logfile, "a"))

target_g = g[500:].mean()
target_o = o[500:].mean()
t = 0

#%%    
if platform == "linux":
    time_limit = env.time_length
else:
    time_limit = 200
    
while t < time_limit:
    
    print("TIME: ", t)
    
    a = mcts.mcts(env, t, net,
                  scl = scl,
                  criterion = criterion,
                  tpa = tpa,
                  tree_horizon = t_horiz,
                  rollout_horizon = r_horiz,
                  n_rolls = n_rolls,
                  gamma = gamma)

    rewards[t] = len(a)
    env.removed_container[t].update(a)
    
    
    t += 1    
    t_run_start = max(0, t-100)
    t_target_stop = min(t, 2000)

    print(" t:", t,
          " Perf: {:1.3f}".format(np.mean(rewards[:t])),
          " G: {:1.3f}".format(np.mean(g[:t])),
          " O: {:1.3f}".format(np.mean(o[:t])),
          file = open(logfile, "a"))
        

    if t > 200 and np.mean(rewards[100:t]) < 3.0:
        system("qsub job_mcts.pbs")
        system("rm -rf MCTSG_{}*".format(name))
        exit()


    if platform == "linux" and t % 100 == 0:
        with open("results/" + prefix + name + ".pkl", "wb") as f:
            pickle.dump(file = f, 
                        obj = {"file": file,
                               "environment": envname,
                               "this_rewards": rewards,
                               "this_matched": matched,
                               "net": net,
                               "opt": o,
                               "greedy": g,
                               "scl": scl,
                               "criterion": criterion,
                               "tpa": tpa,
                               "r_horiz": r_horiz,
                               "t_horiz": t_horiz,
                               "n_rolls": n_rolls,
                               "net_file": net_file,
                               "config": config})


#%%

results = [file,
           net_file,
           envname,
           *config,
           sum(rewards),
           g.sum(),
           o.sum()]


with open("results/mcts_results_greedy.txt", "a") as f:
    s = ",".join([str(s) for s in results])
    f.write(s + "\n")


if platform == "linux":
    from os import system
    system("qsub job_mcts.pbs")
    system("rm -rf MCTS_{}*".format(name))
    exit()


