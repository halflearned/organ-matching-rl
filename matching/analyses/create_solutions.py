#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 18:22:18 2017

@author: vitorhadad
"""

import pickle
from numpy.random import randint
from random import choice

from matching.environment.abo_environment import ABOKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.solver.kidney_solver2 import optimal, greedy

while True:

    try:
        entry_rate = choice([1,2,3,4,5,6,7,8,9,10])
        death_rate = choice([.01,.02, .03, .04, .05,.06, 0.07, 0.08 ,0.09,
                             .1,  .2,  .3,  .4, .5, .6,.7,.8,.9])
        mcl = choice([2, 3])
        envtype = choice(["abo", "saidman", "optn"])
        exchange = {"abo": ABOKidneyExchange,
                    "saidman": SaidmanKidneyExchange,
                    "optn": OPTNKidneyExchange}
        
        print("Entry rate: ", entry_rate)
        print("Death rate: ", death_rate)
        print("Max cycle length: ", mcl)

        filename = "{}_{}_{}_{}_{}.pkl".format(
                                 envtype,
                                 entry_rate,
                                 int(10*death_rate),
                                 mcl,
                                 str(randint(1e8)))

        
        env = exchange[envtype](entry_rate,
                       death_rate,
                       3000)

        opt = optimal(env, max_cycle_length = mcl)
        gre = greedy(env, max_cycle_length = mcl)

        res = [envtype,
               env.entry_rate, env.death_rate, 
               mcl, env.time_length, 
               opt["obj"], gre["obj"],
               gre["obj"]/opt["obj"]]

        pickle.dump({"opt":opt,
                     "greedy":gre,
                     "env":env},
                    file = open("results/" + filename,"wb"))

        with open("results/summary_results.txt", "a") as f:
            f.write(",".join([str(s) for s in res]) + "\n")

    except Exception as e:
        print(e)
        print("Failed with entry_rate: {} and death_rate: {}".format(entry_rate, death_rate))
