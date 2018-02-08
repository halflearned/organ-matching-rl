#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 19:58:30 2017

@author: vitorhadad
"""

import pandas as pd
import matplotlib.pyplot as plt
from os import system

#scp -P 22022 baisihad@sirius.bc.edu:/data/baisihad/matching/results/bandit_results3.txt results/      
             
df = pd.read_csv("results/bandit_results.txt",
                 names = ["algorithm", "param", "thres", "environment",
                  "seed", "time", "entry", "death", "rewards",
                  "greedy", "optimal"])
#%%
df["max_time"] = df.groupby("seed")["time"] \
                    .transform(max)
df = df.query("max_time >= 1000")


df["environment"] = df["environment"].fillna("ABO(5,.1,12345)")
df["env"] = df["environment"].apply(lambda s: s.split("(")[0])

                   
perf = df.groupby(['env', "algorithm", "param", "thres"])\
                    [["rewards","greedy","optimal"]]\
                    .mean()
                    
                    
perf["percentage_gain"] = 100*(perf["rewards"] - perf["greedy"])/perf["greedy"]
