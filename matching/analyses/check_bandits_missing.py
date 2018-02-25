import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

sns.set_style("white")

# scp -P 22022 baisihad@sirius.bc.edu:/data/baisihad/matching/results/bandit_results3.txt results/
#


df = pd.read_csv("/Users/vitorhadad/Documents/kidney/matching/results/bandit_results_Feb14.txt",
                 names=["algorithm", "param", "thres", "environment",
                        "seed", "time", "entry", "death", "rewards", "greedy", "optimal"])

envs = ["ABO", "RSU", "OPTN"]
algos = ["UCB1", "Thompson", "EXP3"]

# Keep only runs between 500 and 1000
df["max_time"] = df.groupby("seed")["time"].transform(max)
df = df.query("max_time >= 700")
df = df.query("time > 500")

# Fill older runs
df["environment"] = df["environment"].fillna("ABO(5,.1,12345)")
df["env"] = df["environment"].apply(lambda s: s.split("(")[0])

# Fill thompson sampling parameters

param = 0.1
thres = 0.5
df["param"] = df["param"].fillna(param)  # So we don't have to deal with thompson separately when plotting

df = df.query("param == 0.1 & thres == 0.5")


idx = pd.MultiIndex\
    .from_product([envs, algos, [3, 5, 7], [5, 8, 10]],
                  names=["environment", "algorithm", "entry", "death"])

max_times = df.groupby(["env", "algorithm", "entry", "death", "seed"])\
                ["time"].max()

completed = df.groupby(["env", "algorithm", "entry", "death"])\
            ["seed"].nunique()\
            .loc[idx].fillna(0)

missing = completed\
            .pipe(lambda x: x < 3)\
            .index.tolist()

print(missing)
