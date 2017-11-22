#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:58:43 2017

@author: vitorhadad
"""

import numpy as np
from random import choice
import pickle

from matching.environment.abo_environment import ABOKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.solver.kidney_solver2 import optimal, greedy


entry_rate = choice([5, 10,   15,   20,  25,  30])
death_rate = choice([1e-10, 0.01, 0.05, 0.1, 0.2])
envclass = choice([ABOKidneyExchange, SaidmanKidneyExchange, OPTNKidneyExchange])
time_length = 1000
burn_in = 0

env = envclass(entry_rate,
               death_rate,
               time_length)

o = optimal(env)
g = greedy(env)

filename = "ovsg_{}.pkl".format(np.random.randint(1e8))

pickle.dump({"env": env,
             "o": o,
             "g": g},
            file = open(filename, "wb"))
             