#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:12:29 2018

@author: vitorhadad
"""

import pandas as pd
import numpy as np
import seaborn as sns

#%%
df = pd.read_csv("results/ucb1mod_results.txt",
        names = ["env_type", "algo", "seed",
                 "time","entry_rate", "death_rate",
                 "n_sims", "constant", "thres", "pmatch",
                 "this","greedy","opt"],
                 header = None).dropna()

#%%
df["max_time"] = df.groupby("seed")["time"].transform(max)

cond = (df["max_time"] > 100) 
    
res = df.loc[cond].groupby(["seed", "entry_rate", "death_rate", "algo", "n_sims", "constant", "thres"])[["this", "greedy", "opt"]].mean()
res["tg_ratio"]  = res["this"]/res["greedy"]
res["to_ratio"]  = res["this"]/res["opt"]
