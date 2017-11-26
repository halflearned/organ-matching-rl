#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7th 17:44:35 2017

MCTS with policy function

"""


import numpy as np

    
from collections import defaultdict
from random import choice
from os import listdir
import torch
from sys import platform
import pickle


from matching.solver.kidney_solver2 import  optimal, greedy
import matching.tree_search.mcts_increasing as mcts
from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.utils.env_utils import snapshot, get_environment_name
from matching.utils.data_utils import get_n_matched

#%%



if platform == "linux":
    scl, criterion = choice([(.1,  "visits"),
                             (.5,   "visits"),
                             (1,    "visits"),
                             (2,    "visits")])
    
    tpa = choice([1,5,10])
    t_horiz = choice([1,2,3,4,5])
    r_horiz = choice([2,5,10,20])
    n_rolls = choice([1,5,10])
    net_file = choice(["RL_23101647",
                       "RL_26213785",
                       "RL_73162545",
                       "RL_74678542",
                       "RL_81274922"])
    
    gamma = choice([1, .95, .9, .75, .5, .1])
    
else: 
    scl = 1
    criterion = "visits"
    tpa = 1
    t_horiz = 1
    r_horiz = 20
    n_rolls = 20
    net_file = "RL_23101647" 
    gamma = 0.9


if net_file is not None:
    net = torch.load("results/" + net_file)
    net.eval()
else:
    net = None


config = (scl, criterion, tpa, n_rolls, 
          t_horiz, r_horiz, net_file, gamma)



name = str(np.random.randint(1e8))      

file = choice([f for f in listdir("results/")
                if f.startswith("optn_")])
    
logfile = "MCTSI_"+ name + ".txt"

data  = pickle.load(open("results/" + file, "rb"))
env = data["env"]
envname = get_environment_name(env)
o = get_n_matched(data["opt_matched"], 0, env.time_length)
g = get_n_matched(data["greedy_matched"], 0, env.time_length)

matched = defaultdict(list)
rewards = np.zeros(env.time_length)


print("scl " + str(scl) + \
      "tpa " + str(tpa) + \
      "t_horiz " + str(t_horiz) + \
      "r_horiz " + str(r_horiz) + \
      "n_rolls " + str(n_rolls) + \
      "net " + str(net_file) + \
      "gamma " + str(gamma),
      file = open(logfile, "a"))

target_g = g[500:2000].mean()
target_o = o[500:2000].mean()
t = 0
#%%    

while t < env.time_length:
    
    a = mcts.mcts(env, t, net = net,
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
          " Run: {:1.3f}".format(np.mean(rewards[t_run_start:t])),
          " Target: {:1.3f}".format(np.mean(rewards[500:t_target_stop])),
          " G: {:1.3f}".format(target_g),
          " O: {:1.3f}".format(target_o),
          file = open(logfile, "a"))
        


    if platform == "linux" and t % 100 == 0:
        with open("results/MCTSI_" + name + ".pkl", "wb") as f:
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


with open("results/mcts_results_increasing.txt", "a") as f:
    s = ",".join([str(s) for s in results])
    f.write(s + "\n")


if platform == "linux":
    from os import system
    system("qsub job_mcts.pbs")
    system("rm -rf MCTS_{}*".format(name))
    exit()



