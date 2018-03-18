"""

scp -P 22022 baisihad@sirius.bc.edu:/data/baisihad/matching/results/greedy_opt_comparison_results_capped.txt /Users/vitorhadad/Documents/kidney/matching/results/

"""

import pandas as pd

df = pd.read_csv("results/capped_results.txt",
                 header=None,
                 names=["env", "seed",
                        "entry", "death", "time_length", "fraction_ndd",
                        "max_cycle", "max_chain",
                        "opt", "greedy", "ratio",
                        "time"])
