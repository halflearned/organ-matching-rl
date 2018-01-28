#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 19:58:30 2017

@author: vitorhadad
"""

import pandas as pd
import matplotlib.pyplot as plt
from os import system

#scp -P 22022 baisihad@sirius.bc.edu:/data/baisihad/matching/results/bandits_results3.txt results/


df = pd.read_csv("results/bandit_results3.txt", 
                   names = ["algorithm", "param", "thres", 
                            "environment","seed","time",
                            "entry_rate", "death_rate", 
                            "rewards", "greedy", "opt"])

df.loc[df["algorithm"]=="UCB1","algorithm"] = df.loc[df["algorithm"]=="UCB1","param"].apply(lambda c: "UCB1(c={})".format(c))

results = df.groupby(["algorithm", "seed"])[["rewards","greedy","opt"]].mean()

by_time = df.groupby(["algorithm", "time"])[["rewards", "greedy", "opt"]].mean().unstack()

r= by_time["rewards"].T
g =by_time["greedy"].T

ratio = (r/g).T

fig, ax = plt.subplots(1, figsize=(10, 5))
ax = ratio.T.rolling(100).mean().plot(ax = ax)
ax.legend(bbox_to_anchor=(1.2, 1.05), fancybox=True, shadow=True)
