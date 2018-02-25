#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 20:37:48 2018

@author: vitorhadad
"""

# SEND
# scp -P 22022 -r matching/bandits/ baisihad@sirius.bc.edu:/data/baisihad/matching/matching/

#%%

from sys import platform, argv
from os import system

from random import choice
import numpy as np
from time import time

from matching.utils.env_utils import two_cycles
from matching.solver.kidney_solver2 import optimal, greedy
from matching.utils.data_utils import clock_seed, get_n_matched

from matching.environment.abo_environment import ABOKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange

from matching.bandits.exp3 import EXP3
from matching.bandits.ucb1 import UCB1
from matching.bandits.thompson import Thompson


t1 = time()

if len(argv) == 1:
    argv = [None, "ABO", "Thompson", 5, 0.1]

environment = argv[1]
algorithm = argv[2]
entry_rate = float(argv[3])
death_rate = float(argv[4])

print("Using {} {} {} {}".format(*argv[1:]))

seed = clock_seed()

max_time = 1001 if platform == "linux" else 5

envs = {"ABO": ABOKidneyExchange,
        "RSU": SaidmanKidneyExchange,
        "OPTN": OPTNKidneyExchange}
env = envs[environment]
thres = choice([.5])
gamma = choice([.1])
c = choice([.1])
n_iters = 1
ipa = 20


for i in range(n_iters):

    env = env(entry_rate, death_rate, max_time)

    opt = optimal(env)
    gre = greedy(env)
    o = get_n_matched(opt["matched"], 0, env.time_length)
    g = get_n_matched(gre["matched"], 0, env.time_length)

    rewards = np.zeros(env.time_length)
    log_every = 1

    np.random.seed(seed)

    for t in range(env.time_length):
        while True:
            cycles = two_cycles(env, t)
            if len(cycles) == 0:
                break
            else:
                if algorithm == "EXP3":
                    algo = EXP3(env, t, gamma=gamma, thres=thres, iters_per_arm=ipa)
                elif algorithm == "Thompson":
                    algo = Thompson(env, t, thres=thres, iters_per_arm=ipa)
                elif algorithm == "UCB1":
                    algo = UCB1(env, t, c=c, thres=thres, iters_per_arm=ipa)

                algo.simulate()
                res = algo.choose()
                if res is not None:
                    env.removed_container[t].update(res)
                    rewards[t] = len(env.removed_container[t])
                else:
                    break

        if t == env.time_length - 1:
            rewards[t] += len(optimal(env, t, t)["matched_pairs"])

        if algorithm == "EXP3":
            param = gamma
        elif algorithm == "Thompson":
            param = np.nan
        elif algorithm == "UCB1":
            param = c


        if t % log_every == 0 and t > 0:
            stats=[algorithm,
                   param,
                   ipa,
                   thres,
                   "\"" + str(env) + "\"",
                   seed,
                   t,
                   int(env.entry_rate),
                   int(env.death_rate*100),
                   rewards[t],
                   g[t],
                   o[t]]
            msg = ",".join(["{}"]*len(stats)).format(*stats)

            if platform == "linux":
                with open("results/bandit_results5.txt", "a") as f:
                    print(msg, file=f)
            else:
                print(t, np.sum(rewards[:t+1]),
                          np.sum(g[:t+1]),
                          np.sum(o[:t+1]))


cmd = 'qsub -F "{} {} {} {}" job_bandits.pbs'.format(*argv[1:])
system(cmd)

t2 = time()

print("Finished {} {} {} {} in {:4.2f} seconds".format(*argv[1:], t2 - t1))