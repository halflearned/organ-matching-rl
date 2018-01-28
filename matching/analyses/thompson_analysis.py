#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:04:29 2017

@author: vitorhadad
"""
import pandas as pd
import numpy as np
import seaborn as sns

#%%
df = pd.read_csv("results/abo_thompson_results2.txt",
        names = ["env_type", "algo", "seed",
                 "time","entry_rate", "death_rate",
                 "n_sims", "n_priors", "thres", "pmatch",
                 "this","greedy","opt"],
                 header = None).dropna()

#%%
df["max_time"] = df.groupby("seed")["time"].transform(max)

cond = (df["max_time"] == 1000) & \
        (df["thres"] == 0.25) & \
        (df["time"] > 200) 
    
res = df.loc[cond].groupby(["seed", "entry_rate", "death_rate", "algo", "n_sims", "n_priors"])[["this", "greedy", "opt"]].mean()
res["tg_ratio"]  = res["this"]/res["greedy"]
res["to_ratio"]  = res["this"]/res["opt"]

print(np.sum(res["tg_ratio"] >= 1))
summary = res.groupby(["algo", "entry_rate", "death_rate", "n_priors", "n_sims"])["tg_ratio"].agg(["mean","size"])

print(summary)
print(res.groupby("algo")["tg_ratio"].mean())

#%%

#df = pd.read_csv("results/abo_thompson_results.txt",
#        names = ["env_type", "algo", "seed",
#                 "time","entry_rate", "death_rate",
#                 "n_sims", "n_priors", "thres",
#                 "this","greedy","opt"],
#                 header = None).dropna()