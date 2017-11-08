#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:11:10 2017

@author: vitorhadad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv("predicting_matched.csv")

df = df[df["sampling"] == "under"]
df["algo_group"] = df["algo"].apply(lambda s: s.split("(")[0])

df["THIS/OPT"] = df["THIS"]/df["OPT"]
grp = df.groupby(["algo_group"])["THIS/OPT"].max()

fig, ax = plt.subplots(1, figsize = (8, 4))
grp.plot(kind = "bar",
         ax = ax)

ax.set_xlabel("Estimator type (using best cross-validated parameters)", fontsize = 12)

g_ratio = df["GREEDY"]/df["OPT"]

ax.bar(left = 5, height = np.mean(g_ratio),
           color = "darkgreen",
           label = "Greedy",
           width = 0.5,
           alpha = .8,
           yerr = np.std(g_ratio)/np.sqrt(len(g_ratio)))

ax.axhline(np.mean(g_ratio),
           -.5, 5.5,
           linestyle = "--",
           color = "black")

ax.set_xlim(-.5, 5.5)

ax.set_xticks(range(6))
ax.set_xticklabels(["Gaussian Process", 
                    "Gradient Boosting",
                    "Logistic Regression",
                    "Random Forests",
                    "Support Vector Machines",
                    "Greedy"],
                   rotation = 45,
                   ha = "right")

ax.set_ylim(0, 1)
ax.set_title("Average performance of traditional estimators relative to OPT",
             fontsize = 14)

fig.savefig("traditional_estimators.png")