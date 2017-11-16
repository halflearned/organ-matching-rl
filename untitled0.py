#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:50:52 2017

@author: vitorhadad
"""

#%%
import matplotlib.pyplot as plt

tm = get_n_matched(matched)
om = get_n_matched(opt["matched"])
gm = get_n_matched(g["matched"])

ts=np.arange(1, env.time_length + 1)

tmc = np.cumsum(tm)/ts
omc = np.cumsum(om)/ts
gmc = np.cumsum(gm)/ts

plt.plot(ts, tmc, color = "green", label = "this")
plt.plot(ts, omc, color = "blue", label = "optimal")
plt.plot(ts, gmc, color = "orange", label = "greedy")
plt.legend(loc="best")
    
