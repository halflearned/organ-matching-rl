#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:48:03 2017

@author: vitorhadad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

df = pd.read_csv("results/summary_results.txt",
                 header = None,
                 names = ["environment", "entry", "death","mcl","time",
                          "greedy", "optimal", "ratio"])

df["entry"] = df["entry"].astype(int)
df["ratio"] *= 100
df["death"] = (df["death"]*100).astype(int)

env_types = ["abo", "saidman", "optn"]
mcls = [2,3]

fig, axs = plt.subplots(3, 2, figsize = (11,12))
cbar_ax = fig.add_axes([.25, .1, .5, .01])

fig.tight_layout(rect = [.15, .15, .9, .9])
fig.subplots_adjust(top = 0.9, hspace = .4)
fig.suptitle("Performance ratio: Myopic / OPT (%)")

axs = axs.flatten()

for k,(env_type,mcl) in enumerate(product(env_types, mcls)):

    cond = (df["environment"] == env_type) & (df["mcl"] == mcl)
    means = df[cond].groupby(["entry", "death"])["ratio"].mean().reset_index()
    tab = means.pivot("entry","death","ratio")
    
    ax = sns.heatmap(tab, vmin = 77, vmax = 100,
                     ax = axs[k],
                     cbar= k == 0,
                     cmap = plt.cm.viridis_r,
                     cbar_ax=None if k else cbar_ax,
                     cbar_kws={"orientation": "horizontal"})
    
    ax.set_xticklabels(tab.columns, 
                       rotation = 45,
                       ha = "center")
        
    if k % 2 == 0:
        ax.set_ylabel("Entry rate", fontsize = 13)
    else:
        ax.set_ylabel("", fontsize = 13)
        
    if k in [4,5]:
        ax.set_xlabel("Death rate $\\times 100$", fontsize = 13)
    else:
        #ax.set_xticklabels([])
        ax.set_xlabel("", fontsize = 13)      
        
        
    ax.set_title("{} (Max cycle length: {})".format(env_type.upper(), mcl), 
                 fontsize = 13)



    
ax.get_figure().savefig(
   "figures/myopic_vs_opt_performance_ratio.pdf".format(env_type))


#%%
from itertools import product
env_type = "optn"
mcl = 2

df = pd.read_csv("results/summary_results.txt",
                 header = None,
                 names = ["environment", "entry", "death","mcl","time",
                          "greedy", "optimal", "ratio"])
df["entry"] = df["entry"].astype(int)
df["ratio"] *= 100
df["death"] = (df["death"]*100).astype(int)
cond = (df["environment"] == env_type) & (df["mcl"] == mcl)
means = df[cond].groupby(["entry", "death"])["ratio"].mean().reset_index()
tab = means.pivot("entry","death","ratio")

null = np.argwhere(tab.isnull().values)
missing_entry = tab.index[null[:,0]]
missing_death = tab.columns[null[:,1]]/100
missing_mcl = [2]*len(missing_entry)

for e,d,m in zip(missing_entry, missing_death, missing_mcl):
    mem = 100#min(int(e*10/2 + 20*(d < 0.08) + 30), 100)
    walltime = 99#nt(e*5) + (int(1/d) // 5)
    cmd = 'qsub -F "{} {} {:1.2f} {}" job_solutions.pbs -l mem={}GB,walltime={}:00:00'\
            .format(env_type,e,d,m,mem,walltime)
    print(cmd)
