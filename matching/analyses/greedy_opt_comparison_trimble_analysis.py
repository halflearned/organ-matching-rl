"""

Feb 28th
scp -P 22022 baisihad@sirius.bc.edu:/data/baisihad/matching/results/greedy_opt_comparison_results_trimble.txt /Users/vitorhadad/Documents/kidney/matching/results

"""


import pandas as pd
import numpy as np
from statsmodels.api import OLS

path = "/Users/vitorhadad/Documents/kidney/matching/results/"

#
df = pd.read_csv(path + "greedy_opt_comparison_results_trimble.txt",
                 header=None,
                 names=["environment", "entry", "death", "max_time", "fraction_ndd",
                        "max_cycle", "max_chain",
                        "greedy", "optimal", "ratio", "elapsed"])

config = ["entry", "death", "max_cycle", "max_chain", "fraction_ndd"]
config2 = ["entry", "death", "max_cycle", "fraction_ndd"]

ols = OLS(endog=df["ratio"], exog=df[config])

print(ols.fit().summary())

tab = df.groupby(config)["ratio"].agg(["mean", "sem", "size"])
for c in [2, 3]:
    for e in df["entry"].unique():
        for d in df["death"].unique():
            try:
                diff = tab.xs((e, d, c, 1)) - tab.xs((e, d, c, 0))
                print(e, d, c, np.mean(diff["mean"]))
                tab.xs((e, d, c, 0.05), level=config2)["mean"].plot()
            except KeyError:
                pass
plt.show()

