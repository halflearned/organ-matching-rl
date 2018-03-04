#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:48:03 2017

@author: vitorhadad
"""

from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("/Users/vitorhadad/Documents/kidney/matching/results/greedy_opt_comparison_results.txt",
                 header=None,
                 names=["environment", "entry", "death", "time", "mcl",
                        "greedy", "optimal", "ratio"])

df["entry"] = df["entry"].astype(int)
df["ratio"] *= 100
# df["death"] = (df["death"]*100).astype(int)

env_types = ["ABO", "RSU", "OPTN"]
mcls = [2, 3]

fig, axs = plt.subplots(3, 2, figsize=(11, 12))
cbar_ax = fig.add_axes([.25, .1, .5, .01])

fig.tight_layout(rect=[.15, .15, .9, .9])
fig.subplots_adjust(top=0.9, hspace=.4)
fig.suptitle("Performance ratio: Myopic / OPT (%)")

axs = axs.flatten()

for k, (env_type, mcl) in enumerate(product(env_types, mcls)):

    cond = (df["environment"] == env_type) & (df["mcl"] == mcl)
    means = df[cond].groupby(["entry", "death"])["ratio"].mean().reset_index()
    tab = means.pivot("entry", "death", "ratio")

    ax = sns.heatmap(tab, vmin=77, vmax=100,
                     ax=axs[k],
                     cbar=k == 0,
                     cmap="cubehelix_r",
                     cbar_ax=None if k else cbar_ax,
                     cbar_kws={"orientation": "horizontal"})

    ax.set_xticklabels(tab.columns,
                       rotation=45,
                       ha="center")

    if k % 2 == 0:
        ax.set_ylabel("Entry rate", fontsize=13)
    else:
        ax.set_ylabel("", fontsize=13)

    if k in [4, 5]:
        ax.set_xlabel("Death rate $\\times 100$", fontsize=13)
    else:
        # ax.set_xticklabels([])
        ax.set_xlabel("", fontsize=13)

    ax.set_title("{} (Max cycle length: {})".format(env_type.upper(), mcl),
                 fontsize=13)

ax.get_figure().savefig(
    "/Users/vitorhadad/Documents/kidney/matching/phd_thesis/figures/greedy_opt_comparison_2.pdf".format(env_type))

# %%


env_types = ["ABO", "RSU", "OPTN"]
mcls = [2]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
cbar_ax = fig.add_axes([.9, .3, .015, 0.5])
cbar_ax.tick_params(labelsize=15)

fig.tight_layout(rect=[.15, .15, .9, .9])
fig.subplots_adjust(top=0.85, hspace=.4)
fig.suptitle("Performance ratio: Myopic / OPT (%)", fontsize=18)

axs = axs.flatten()

for k, (env_type, mcl) in enumerate(product(env_types, mcls)):
    cond = (df["environment"] == env_type) & (df["mcl"] == mcl)
    means = df[cond].groupby(["entry", "death"])["ratio"].mean().reset_index()
    tab = means.pivot("entry", "death", "ratio")

    ax = sns.heatmap(tab, vmin=83, vmax=100,
                     ax=axs[k],
                     cbar=k == 2,
                     cmap="cubehelix_r",
                     cbar_ax=None if k < 2 else cbar_ax)

    ax.set_xticklabels(tab.columns,
                       fontsize=13,
                       rotation=45,
                       ha="center")

    ax.set_yticklabels(reversed(tab.index),
                       fontsize=13)

    ax.set_ylabel("Entry rate", fontsize=14)
    ax.set_xlabel("Death rate $\\times 100$", fontsize=14)

    ax.set_title("{}".format(env_type.upper(), mcl),
                 fontsize=14)

ax.get_figure().savefig(
    "/Users/vitorhadad/Documents/kidney/matching/phd_thesis/figures/greedy_opt_comparison_mcl2_2.pdf".format(env_type))
