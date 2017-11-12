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

df = pd.read_csv("results/mcts_with_opt_rollout_results.txt",
                 header = None,
                 names = ["seed", "er","dr","time_length",
                          "scl","tpa","n_rolls","t_horiz",
                          "r_horiz","gcn1","gcn2",
                          "use_priors", 
                          "this","greedy","opt"]).dropna()


df["r_horiz"] = 10
df2 = pd.read_csv("results/mcts_with_opt_rollout_results2.txt",
                 header = None,
                 names = ["seed", "er","dr","time_length",
                          "scl","tpa","n_rolls","t_horiz",
                          "r_horiz","gcn1","gcn2",
                          "use_priors", 
                          "this","greedy","opt"]).dropna()

df = pd.concat([df, df2], 0).reset_index()


def clean_gcn(s):
    x = str(s).replace("(","").replace(")","")
    return float(x)

df["use_priors"] = None

params = [
      "scl","tpa","t_horiz",
      "r_horiz"]

df["this_g_ratio"] = df["this"]/df["greedy"]
df["this_opt_ratio"] = df["this"]/df["opt"]
df["greedy_opt_ratio"] = df["greedy"]/df["opt"]


tab = df.groupby(params)["this_g_ratio"].agg(["mean", "size"])
print(tab.sort_values("mean", ascending = False))
#%%
for p in params:
    print(df.groupby(p)["this_g_ratio"].median())
    plt.figure()
    plt.scatter(df[p], df["this_opt_ratio"])
    plt.title(p)
    
    
#%%
s = sm.OLS(exog = sm.add_constant(df[["scl","tpa", "t_horiz","r_horiz", "greedy_opt_ratio"]]),
           endog = df["this_opt_ratio"])

print(s.fit().summary())