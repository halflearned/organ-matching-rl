from random import choice

import pandas as pd

df = pd.read_csv("results/"
                 "capped_results.txt",
                 names=["environment", "seed",
                        "entry", "death", "time_length", "fraction_ndd",
                        "max_cycle", "max_chain",
                        "opt", "greedy", "ratio",
                        "elapsed"])

params = ["environment",
          "entry", "death", "fraction_ndd",
          "max_cycle", "max_chain"]

opt = df.groupby(params)["opt"].mean()
greedy = df.groupby(params)["greedy"].mean()
ratio = df.groupby(params)["ratio"].mean()

idx = pd.MultiIndex.from_product([["ABO", "RSU", "OPTN"],
                                  [3, 5, 7],
                                  [0.5, 0.25, 0.1, 0.075, 0.050],
                                  [0.05, 0.1],
                                  [0, 2, 3],
                                  [0, 2, 3, 4]],
                                 names=["environment", "entry",
                                        "death", "fraction_ndd",
                                        "max_cycle", "max_chain"])

complete = ratio.reindex(idx)

missing = complete[complete.isnull()] \
    .reset_index()[params] \
    .values.tolist()


for args in idx.tolist():
    if args[0] == "OPTN" and \
        args[2] < 0.5 and args[3] == 0.05 and \
        (args[4] > 0 or args[5] > 0):

        cmd = 'qsub -F "{} {} {} {} {} {}" job_comparison.pbs' \
                .format(*args)
        print(cmd)
