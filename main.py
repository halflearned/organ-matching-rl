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


scl, criterion = choice([(.1,  "visits"),
                         (.5,   "visits"),
                         (1,    "visits"),
                         (2,    "visits")])

tpa = choice([5, 10])
t_horiz = choice([5, 22])
r_horiz = choice([10, 45])
n_rolls = choice([1, 5])
net_file = choice(["RL_18951235",
                   "RL_23101647",
                   "RL_26213785",
                   "RL_28123910",
                   "RL_73162545",
                   "RL_74678542",
                   "RL_80663654",
                   "RL_81274922"])

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

o = get_n_matched(data["opt_matched"], 0, env.time_length)
g = get_n_matched(data["greedy_matched"], 0, env.time_length)

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
    
    
    print(" Time:", t,
          " Action:",a,
          " R:", sum(rewards),
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
        

    if t > 200 and np.sum(rewards) < (0.85*o[:t].sum()):
        from os import system
        system("qsub job_mcts.pbs")
        system("rm -rf MCTS_{}*".format(name))
        exit()
        


    if platform == "linux" and t % 100 == 0:
        envname = get_environment_name(env)
        with open("results/" + name + ".pkl", "wb") as f:
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
print("MCTS loss: ", sum(rewards))
print("GREEDY loss:",  g.sum())
print("OPT loss:", o.sum())



results = [file,
           net_file,
           envname,
           *config,
           sum(rewards),
           g.sum(),
           o.sum()]


with open("results/mcts_results9.txt", "a") as f:
    s = ",".join([str(s) for s in results])
    f.write(s + "\n")





