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

df = pd.read_csv("results/mcts_results2.txt",
                 header = None,
                 names = ["environment", "seed", "er","dr",
                          "time_length",
                          "scl","criterion","tpa","n_rolls",
                          "t_horiz",
                          "r_horiz","net_file", 
                          "this","greedy","opt"])


df= df[df["time_length"] > 50]

params = ["scl","tpa","n_rolls", "t_horiz","r_horiz"]

df["this_g_ratio"] = df["this"] - df["greedy"]
df["this_opt_ratio"] = df["this"] - df["opt"]
df["greedy_opt_ratio"] = df["greedy"] - df["opt"]


tab = df.groupby(params)["this_g_ratio"].agg(["mean", "size"])

