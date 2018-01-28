#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 20:00:37 2017

@author: vitorhadad
"""

import pickle
import numpy as np
import pandas as pd
from os import listdir
from sys import argv

from matching.environment.optn_environment import OPTNKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.environment.abo_environment import ABOKidneyExchange

from matching.solver.kidney_solver2 import optimal_with_discount, optimal, greedy
from matching.utils.data_utils import  clock_seed, cumavg, get_size, get_n_matched
import matching.utils.data_utils as utils


outfile = argv[1]
infiles = argv[2:]

for f in infiles:
    
    df = pickle.load(open("results/"+ f, "rb"))
    
    mo = utils.get_n_matched(df["opt"]["matched"], 0, df["env"].time_length)[1000:].mean()
    mg = utils.get_n_matched(df["greedy"]["matched"], 0, df["env"].time_length)[1000:].mean()
    so = utils.get_size(df["env"], df["opt"]["matched"])[1000:].mean(),
    sg = utils.get_size(df["env"], df["greedy"]["matched"])[1000:].mean(),

    data = [df["env"].entry_rate,
            df["env"].death_rate,
            df["env"].time_length,
            mo,
            mg,
            so,
            sg,
            df["opt"]["obj"],
            df["greedy"]["obj"],
            df["greedy"]["obj"]/df["opt"]["obj"]]
    
    with open("results/" + outfile, "rb") as outf:
        outf.write(",".join([str(s) for s in data]) + "\n")
    
    

