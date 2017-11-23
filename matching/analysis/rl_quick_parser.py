#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:56:48 2017

@author: vitorhadad
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

# RL_73162545
# RL_73438689

r = []
g = []
o = []
with open("results/RL_73162545.txt", "r") as file:
    for line in file:
        numbers=re.findall("[0-9|\.]+", line)
        r.append(float(numbers[-3]))
        g.append(float(numbers[-2]))
        o.append(float(numbers[-1]))
        

#%%
fig, ax = plt.subplots(1)
pd.Series(r[1000:]).plot(ax = ax, color = "green")
pd.Series(g[1000:]).plot(ax = ax, color = "orange")
pd.Series(o[1000:]).plot(ax = ax, color = "blue")
ax.set_ylim((4,5))


burnin = 500
mr = r[burnin:].mean()
mo = o[burnin:].mean()
mg = g[burnin:].mean()

print(mr, mo, mg, mr/mg - 1)



#%%
r = []
g = []
o = []
with open("results/policy_results.txt", "r") as file:
    for line in file:
        numbers=re.findall("[0-9|\.]+", line)
        r.append(float(numbers[-3]))
        g.append(float(numbers[-2]))
        o.append(float(numbers[-1]))

rg = [(rr, gg, rr/gg) for rr,gg in zip(r,g) if rr > gg]

