"""

Feb 28th
scp -P 22022 baisihad@sirius.bc.edu:/data/baisihad/matching/results/greedy_opt_comparison_results_trimble.txt /Users/vitorhadad/Documents/kidney/matching/results

"""


import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt

path = "/Users/vitorhadad/Documents/kidney/matching/results/"

#
df = pd.read_csv(path + "greedy_opt_comparison_results_trimble.txt",
                 header=None,
                 names=["environment", "entry", "death", "max_time", "fraction_ndd",
                        "max_cycle", "max_chain",
                        "greedy", "optimal", "ratio", "elapsed"])

config = ["environment", "entry", "death", "max_cycle", "max_chain", "fraction_ndd"]

summary = df.groupby(config)["ratio"].agg(["mean", "std", "size"])
abo = summary.xs("ABO")["mean"].unstack("fraction_ndd")
rsu = summary.xs("RSU")["mean"].unstack("fraction_ndd")
optn = summary.xs("OPTN")["mean"].unstack("fraction_ndd")
