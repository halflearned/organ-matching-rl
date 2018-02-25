#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 19:58:30 2017

@author: vitorhadad
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from scipy.stats import ttest_rel
sns.set_style("white")

#scp -P 22022 baisihad@sirius.bc.edu:/data/baisihad/matching/results/bandit_results3.txt results/      
#


df = pd.read_csv("/Users/vitorhadad/Documents/kidney/matching/results/bandit_results_Feb15.txt",
                 names=["algorithm", "param", "thres",
                        "environment", "seed", "time",
                        "entry", "death", "rewards",
                        "greedy", "optimal"])

envs = ["ABO", "RSU", "OPTN"]
algos = ["UCB1", "Thompson", "EXP3"]

# Keep only runs between 500 and 1000
df["max_time"] = df.groupby("seed")["time"] \
                    .transform(max)
df = df.query("max_time >= 250")
df = df.query("time < 1000")
df = df.query("entry < 10")


# Fill older runs
df["environment"] = df["environment"].fillna("ABO(5,.1,12345)")
df["env"] = df["environment"].apply(lambda s: s.split("(")[0])

# Fill thompson sampling parameters

param = 0.1
thres = 0.5
df["param"] = df["param"].fillna(param) # So we don't have to deal with thompson separately when plotting
df = df.query("param == 0.1 & thres == 0.5")

missing = []
# HEATMAPS
for env in envs:

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    cbar_ax = fig.add_axes([.9, .3, .015, 0.5])
    cbar_ax.tick_params(labelsize=15)

    fig.tight_layout(rect=[.15, .15, .9, .9])
    fig.subplots_adjust(top=0.85, hspace=.4)
    fig.suptitle("Environment {}".format(env), fontsize=18)

    for k, algo in enumerate(algos):
        cond = (df["env"] == env) & \
               (df["algorithm"] == algo)

        # Average matching sizes
        avg_matched = df[cond].groupby(["seed", "entry", "death"])[["rewards", "greedy", "optimal"]].mean()

        # Ratio of average matching sizes between this and greedy
        ratio = avg_matched["rewards"] / avg_matched["greedy"]
        avg_ratio = (ratio.groupby(level=["entry", "death"]).mean() - 1) * 100

        diff = avg_matched["rewards"] - avg_matched["greedy"]
        avg_diff = diff.groupby(level=["entry", "death"]).agg(["mean", "sem"])

        tab = avg_ratio.reset_index().pivot("entry", "death", 0)

        # Get missing indices
        idx = pd.MultiIndex.from_product([[3, 5, 7], [5, 8, 10]], names=["entry", "death"])
        miss = tab.stack().reindex(idx) \
            .pipe(lambda x: x[x.isnull()]) \
            .reset_index()[["entry", "death"]] \
            .values.tolist()

        for m in miss:
            missing.append((env, algo, m[0], m[1]/100))


        ax = sns.heatmap(tab, vmin=0, vmax=5,
                         ax=axs[k],
                         cbar=k == 2,
                         cmap="cubehelix_r",
                         cbar_ax=None if k < 2 else cbar_ax,
                         linecolor="black",
                         linewidth=2)

        ax.set_xticklabels(tab.columns.astype(int),
                           fontsize=13,
                           rotation=45,
                           ha="center")

        ax.set_yticklabels(reversed(tab.index.astype(int)),
                           fontsize=13)

        ax.set_ylabel("Entry rate", fontsize=14)
        ax.set_xlabel("Death rate $\\times 100$", fontsize=14)

        ax.set_title("{}".format(algo), fontsize=14)

    plt.show()


    fig.savefig("/Users/vitorhadad/Documents/kidney/matching/phd_thesis/figures/mab_{}.pdf".format(env))


print(missing)

# #####
# TABLES
# #####

# Average per run
avg = df.groupby(["env", "algorithm", "entry", "death"])[["rewards", "greedy"]].agg(["mean", "sem"])
size = df.groupby(["env", "algorithm", "entry", "death"])["seed"].nunique()

# P-values
values = df.groupby(["env", "algorithm", "entry", "death", "seed"])[["rewards", "greedy"]].mean()
pvalues = []
for key in avg.index.tolist():
    m = values.xs(key)
    p = ttest_rel(m["rewards"], m["greedy"]).pvalue
    pvalues.append(p)


avg["diff"] = avg["rewards"]["mean"] - avg["greedy"]["mean"]
avg["pvalue"] = pvalues
avg["ratio"] = 100 * (avg["rewards"]["mean"] / avg["greedy"]["mean"] - 1)
avg["n"] = size

for key in algos:

    table = avg.xs(key, level="algorithm")
    table.columns.set_levels([key, 'Myopic', 'Difference', 'p-value', 'Ratio (%)', 'N'],
                             level=0, inplace=True)
    table.columns.set_levels(['Mean', 'Std Error', ''],
                             level=1, inplace=True)
    table.index.set_levels(["ABO", "OPTN", "RSU"],
                           level=0, inplace=True)
    table.index.name = "Environment"
    table.index.rename(["Environ.", "Entry", "Death"], inplace=True)
    table.to_latex("/Users/vitorhadad/Documents/kidney/matching/phd_thesis/tables/" \
                   "mab_{}.tex".format(key),
                   escape=True,
                   float_format=lambda x: "{:1.3f}".format(x))





