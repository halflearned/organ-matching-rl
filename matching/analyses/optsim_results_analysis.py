#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 19:58:30 2017

@author: vitorhadad
"""

import pandas as pd

df = pd.read_csv("results/bandit_results.txt", 
                   names = ["algorithm","environment","seed","time",
                            "entry_rate", "death_rate", 
                            "rewards", "greedy", "opt"])

results = df.groupby(["algorithm", "seed"])[["rewards","greedy","opt"]].mean()

