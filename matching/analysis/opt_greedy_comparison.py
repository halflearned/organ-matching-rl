#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:35:03 2017

Analyzing several aspects of data

@author: vitorhadad
"""

import numpy as np
from matching.utils.data_utils import merge_data
import matplotlib.pyplot as plt
import pandas as pd


#%% Get data
datasets = merge_data("X", "t", "A", "G", "N", "opt_n_matched", "greedy_n_matched", path = "data/")
X_shape = datasets[0][0]["X"].shape[1]
N_shape = datasets[0][0]["N"].shape[1]
G_shape = datasets[0][0]["G"].shape[1]

n = 190

#%%
# Flattening
XXs = []
As = []
ts = []
ys = []
gs = []
for ds in datasets:
    for item in ds:
        t = item["t"]
        if t == 0:
            XXs.append(np.hstack([item["X"], item["G"], item["N"]]))
            ys.append(item["opt_n_matched"][:n])
            gs.append(item["greedy_n_matched"][:n])
            
#%% Plotting
avg_ys = np.vstack(ys).mean(0)
avg_gs = np.vstack(gs).mean(0)

fig, ax = plt.subplots(1, 2, figsize = (10, 4))

ax[0].plot(avg_ys, label = "optimal")
ax[0].plot(avg_gs, label = "greedy")

ax[1].plot(np.cumsum(avg_ys) / np.arange(n))
ax[1].plot(np.cumsum(avg_gs) / np.arange(n))








