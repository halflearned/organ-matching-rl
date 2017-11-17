#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:09:59 2017

@author: vitorhadad
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("results/mcts_results.txt",
                 header = None,
                 names = ["environment", "seed", "er","dr","time_length",
                          "scl","tpa","n_rolls","t_horiz",
                          "r_horiz","net_file", 
                          "this","greedy","opt"])



params = ["scl","tpa","n_rolls", "t_horiz","r_horiz"]

df["this_g_ratio"] = df["this"] - df["greedy"]
df["this_opt_ratio"] = df["this"] - df["opt"]
df["greedy_opt_ratio"] = df["greedy"] - df["opt"]


tab = df.groupby(params)["this_g_ratio"].agg(["mean", "size"])

print(tab.sort_values("mean", ascending = False))
#%%
for p in params:
    plt.figure()
    plt.scatter(df[p], df["this_g_ratio"])
    plt.title(p)
    
    
#%%
s = sm.OLS(exog = sm.add_constant(
        df[["scl","tpa", "t_horiz","r_horiz", "n_rolls", "greedy_opt_ratio"]]),
           endog = df["this_g_ratio"])

print(s.fit().summary())