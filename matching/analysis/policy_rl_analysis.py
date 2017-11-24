#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 21:02:11 2017

@author: vitorhadad
"""

import pandas as pd


df = pd.read_csv("/Users/vitorhadad/Documents/kidney/matching/results/policy_rl_results.txt",
                 header = None,
                 sep = " ",
                 names = ["seed", "filenumber", "r", "g","o"])

df["rg"] = df["r"]/df["g"]

good = df.groupby("seed")["rg"].transform(lambda x: np.all(x > .9)).astype(bool)
num = df.groupby("seed")["rg"].transform("size")

df = df.loc[(good) & (num > 2)]

df.groupby("seed")["rg"].plot()

