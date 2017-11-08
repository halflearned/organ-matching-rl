#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:29:00 2017

@author: vitorhadad
"""

import pandas as pd
import numpy as np

df = pd.read_csv("adv_vf_results2.txt",
                 names = ["seed",
                          "er","dr","mxd",
                          "hors","scl","prl","tpa","time_length",
                          "this","greedy","opt"])\
                .set_index(["mxd","hors","scl","tpa", "prl"])

df["g_ratio"] = df["greedy"]/df["opt"]
df["this_ratio"] = df["this"]/df["opt"]
df["this_g_ratio"] = df["this"]/df["greedy"]

grp = df.groupby(level = ["mxd","hors","scl","tpa", "prl"])[["this","greedy","opt","g_ratio","this_g_ratio"]]
m = grp.mean()
s = grp.std()
n = grp.size()
n.name = "size"
tab = pd.concat([m, n], axis = 1)
print(tab.round(1))

weakly = np.mean(m["this"] >= m["greedy"])
strictly = np.mean(m["this"] > m["greedy"])

print("MCTS (weakly) beats greedy {:1.2%} of the configurations"\
      .format(weakly))

print("MCTS (strictly) beats greedy {:1.2%} of the configurations"\
      .format(strictly))

this_best = m["this_g_ratio"].argmax()
g_best = m["g_ratio"].argmax()
print("Best OPT ratio:\n", tab.loc[this_best], "\n")
print("Best Greedy ratio:\n", tab.loc[g_best], "\n")

this_g_ratio = 100*(df.loc[this_best, "this"]/df.loc[this_best, "greedy"] - 1)
ax = this_g_ratio.hist()
ax.set_xlim(-10, 10)
ax.set_title("This vs. Greedy Performance Comparison (%)",
             fontsize = 14)

