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
import matching.tree_search.mcts as mcts
from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.utils.env_utils import snapshot, get_environment_name
from matching.utils.data_utils import get_n_matched

#%%


scl, criterion = choice([(.0001,   "visits"),
                         (.1,  "visits"),
                         (.5,   "visits"),
                         (1,    "visits"),
                         (2,    "visits"),
                         (5,    "visits"),
                         (None, "rewards")])

tpa = choice([1, 3, 5])
t_horiz = choice([1, 5])
r_horiz = choice([10, 22, 45])
n_rolls = choice([1])
net_file = "RL_12282732" #choice(net_files)
burnin = 0    

print("USING:")
print("scl", scl)
print("tpa", tpa)
print("t_horiz", t_horiz)
print("r_horiz", r_horiz)
print("n_rolls", n_rolls)
print("net", net_file)


config = (scl, criterion, tpa, n_rolls, t_horiz, r_horiz, net_file)

if net_file is not None:
    net = torch.load("results/" + net_file)
    net.eval()
else:
    net= None


name = str(np.random.randint(1e8))      

file = choice([f for f in listdir("results/")
                if f.startswith("optn_")])
logfile = "MCTS_"+ name + ".txt"

data  = pickle.load(open("results/" + file, "rb"))
env = data["env"]
env.removed_container = data["opt_matched"]
o = get_n_matched(data["opt_matched"], env.time_length)
g = get_n_matched(data["greedy_matched"], env.time_length)

if burnin > env.time_length:
    raise ValueError("Burnin > T")

matched = defaultdict(list)
rewards = []   
t = 0

#%%    

while t < env.time_length:
    
    a = mcts.mcts(env, t, net,
                  scl = scl,
                  tpa = tpa,
                  tree_horizon = t_horiz,
                  rollout_horizon = r_horiz,
                  n_rolls = n_rolls)
    
    
    print(" File", file.split("_")[1],
          " Time:",t,
          " Total:", sum(rewards[:t]),
          " G:", g[:t].sum(),
          " O:", o[:t].sum(),
          file = open(logfile, "a"))
    
    if a is not None:
        
        print("Staying at t.")
        assert a[0] not in env.removed(t)
        assert a[1] not in env.removed(t)
        env.removed_container[t].update(a)
        matched[t].extend(a)
        rewards.append(len(a))
    
    else:
        print("\nDone with", t, ". Moving on to next period\n")
        t += 1


    if platform == "linux" and t % 100 == 0:
        envname = get_environment_name(env)
        with open("results/" + name + ".pkl", "wb") as f:
            pickle.dump(file = f, 
                        obj = {"file": file,
                           "environment": envname,
                           "this_rewards": rewards,
                           "this_matched": matched,
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
print("MCTS loss: ", rewards)
print("GREEDY loss:",  g[:t].sum())
print("OPT loss:", o[:t].sum())



results = [file,
           envname,
           *config,
           rewards,
           g[:t].sum(),
           o[:t].sum()]


with open("results/mcts_results9.txt", "a") as f:
    s = ",".join([str(s) for s in results])
    f.write(s + "\n")





