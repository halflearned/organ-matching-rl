import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import numpy as np
from scipy.stats import ttest_rel

sns.set_style("white")

df = pd.read_csv("/Users/vitorhadad/Documents/kidney/matching/results/traditional_ml_mdp_results_Feb15.txt",
                 names=["environment", "algorithm", "thres", "add",
                        "seed", "time", "entry", "death",
                        "rewards", "greedy", "optimal"])

# Eliminate invalid records
valid = df["environment"].isin(['abo', 'optn', 'saidman']) & \
        df["entry"].isin([5.0, 7.0, 3.0, '5', '3', '7']) & \
        df["death"].isin([10, 8, 5])

df = df.loc[valid]
for c in ["time", "death", "entry", "greedy", "rewards", "optimal", "thres"]:
    df[c] = pd.to_numeric(df[c], errors=np.nan)

df = df.dropna()
for c in ["seed", "time", "death", "entry", "greedy", "rewards", "optimal"]:
    df[c] = df[c].astype(int)
df["thres"] = df["thres"].astype(float)

# Recode some of the variables
# df["env"] = df["environment"].map({"abo": "ABO", "optn": "OPTN", "saidman": "RSU"})
# df["algorithm"] = df["algorithm"].map({"lr": "Logistic", "rf": "Forest", "grb": "Boosting"})
df["env"] = df["environment"]
envmap = {"abo": "ABO", "optn": "OPTN", "saidman": "RSU"}
algomap = {"lr": "Logistic", "rf": "Forest", "grb": "Boosting"}
algomap_long = {"lr": "Logistic Regression",
                "rf": "Random Forests",
                "grb": "Gradient Boosting"}

# Keep only runs between 250 and 1000
df["max_time"] = df.groupby("seed")["time"].transform(max)
df = df.query("max_time >= 990 & time > 250 & thres == 0.3")
df = df[df["add"].isin(["none", "networkx"])]


envs = ["abo", "saidman", "optn"]
algos = ["lr", "rf", "grb"]
adds = ["none", "networkx"]

idx = pd.MultiIndex \
    .from_product([envs, algos, ["none", "networkx"], [3, 5, 7], [5, 8, 10]],
                  names=["env", "algorithm", "add", "entry", "death"])



max_times = df.groupby(["env", "algorithm", "add", "entry", "death", "seed"])["time"].max()
completed = df.groupby(["env", "algorithm", "add", "entry", "death"])["seed"].nunique().loc[idx].fillna(0)
missing = completed.pipe(lambda x: x[x < 5]).index.tolist()

avg = df.groupby(["env", "algorithm", "add", "entry", "death"])[["rewards", "greedy"]].agg(["mean", "sem"])
size = df.groupby(["env", "algorithm", "add", "entry", "death"])["seed"].nunique()

values = df.groupby(["env", "algorithm", "add", "entry", "death", "seed"])[["rewards", "greedy"]].mean()
pvalues = []
for key in avg.index.tolist():
    m = values.xs(key)
    p = ttest_rel(m["rewards"], m["greedy"]).pvalue
    pvalues.append(p)



# print(missing)
avg["diff"] = avg["rewards"]["mean"] - avg["greedy"]["mean"]
avg["pvalue"] = pvalues
avg["ratio"] = 100 * (avg["rewards"]["mean"] / avg["greedy"]["mean"] - 1)
avg["n"] = size

for key in product(algos, adds):
    algo_name = algomap[key[0]]
    table = avg.xs(key, level=["algorithm", "add"])
    table.columns.set_levels([algo_name, 'Myopic', 'Difference', 'p-value', 'Ratio (%)', 'N'],
                             level=0, inplace=True)
    table.columns.set_levels(['Mean', 'Std Error', ''],
                             level=1, inplace=True)
    table.index.set_levels(["ABO", "OPTN", "RSU"],
                           level=0, inplace=True)
    table.index.name = "Environment"
    table.index.rename(["Environ.", "Entry", "Death"], inplace=True)
    table.to_latex("/Users/vitorhadad/Documents/kidney/matching/phd_thesis/tables/"\
                   "traditional_mdp_{}_{}.tex".format(*key),
                   escape=True,
                   float_format=lambda x: "{:1.3f}".format(x))






for algo in algos:

    fig, ax = plt.subplots(2, 3, figsize=(8, 4))
    fig.suptitle("Performance Ratio: {} / Myopic (%)".format(algomap_long[algo]))
    fig.subplots_adjust(hspace=.6)

    cbar_ax = fig.add_axes([.9, .3, .015, 0.5])
    cbar_ax.tick_params(labelsize=15)


    for add in adds:

        for k, env in enumerate(envs):

            for i, add in enumerate(["none", "networkx"]):

                use_cbar = (i == 1) & (k == 2)
                table = avg['Ratio (%)'].xs((env, algo, add)).unstack()
                table = table.clip(lower=-8)
                axx = sns.heatmap(table, ax=ax[i, k],
                            center=0,
                            square=True,
                            vmin=-10,
                            vmax=10,
                            cbar=use_cbar,
                            cbar_ax=cbar_ax if use_cbar else None,
                            cmap="cubehelix_r",
                            linewidth=1)

                this_cbar = axx.collections[0].colorbar
                if this_cbar:
                    this_cbar.set_ticks(range(-10, 10, 2))
                    labels = ["${}$".format(str(s)) for s in range(-10, 10, 2)]
                    labels[0] = "<-10"
                    this_cbar.set_ticklabels(labels)

                ax[i, k].set_xlabel("Death rate ($\\times 10$)")
                ax[i, k].set_ylabel("Entry rate")

                title = "{} {}".format(envmap[env],
                                       "" if add == "none" else "(augmented)")
                ax[i, k].set_title(title)

    fig.savefig("/Users/vitorhadad/Documents/kidney/matching/phd_thesis/figures/"\
                "traditional_ml_mdp_algo_{}.pdf".format(algo))
    plt.show()

