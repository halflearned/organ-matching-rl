#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:07:06 2017

@author: vitorhadad
"""

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


import matching.tree_search.mcts_supergreedy as mcts
from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.utils.env_utils import  get_environment_name
from matching.utils.data_utils import get_n_matched

#%%


if platform == "linux":
    criterion = "rewards"
    scl = 0
    tpa = choice([1])
    t_horiz = choice([1])
    r_horiz = choice([10])
    n_rolls = choice([1, 10, 50, 100, 200, 500])
    net_file = None
    gamma = choice([1, .9, .75, .5, .1])
    file = 'optn_51332267_.pkl'
    
else: 
    scl = .1
    criterion = "visits"
    tpa = 1
    t_horiz = 1
    r_horiz = 10
    net_file = None
    n_rolls = 100
    gamma = .9 
    file = 'sp_optn_12345_.pkl'
    


if net_file is not None:
    net = torch.load("results/" + net_file)
    net.eval()
else:
    net = None


config = (scl, criterion, tpa, n_rolls, 
          t_horiz, r_horiz, net_file, gamma)



name = str(np.random.randint(1e8))      

prefix = "MCTSSG2_"    

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

t = 0

#%%    
if platform == "linux":
    time_limit = 1000#env.time_length
else:
    time_limit = 100
    
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
    
    

    t_run_start = max(0, t-100)
    t_target_stop = min(t, 2000)

    s = " t: {}".format(t) + \
          " Perf: {:1.3f}".format(np.mean(rewards[:t])) + \
          " G: {:1.3f}".format(np.mean(g[:t])) + \
          " O: {:1.3f}".format(np.mean(o[:t])) 

    print(s)
    print(s, file = open(logfile, "a"))
        

    if t > 200 and np.mean(rewards[100:t]) < 3.0:
        system("qsub job_mcts.pbs")
        system("rm -rf {}{}*".format(prefix, name))
        exit()


    if platform == "linux" and t % 100 == 0 and t> 0:
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

    t += 1    
#%%

results = [file,
           net_file,
           envname,
           *config,
           sum(rewards),
           g.sum(),
           o.sum()]


with open("results/mcts_results_supergreedy.txt", "a") as f:
    s = ",".join([str(s) for s in results])
    f.write(s + "\n")


if platform == "linux":
    from os import system
    system("qsub job_mcts.pbs")
    system("rm -rf {}{}*".format(prefix, name))
    exit()


