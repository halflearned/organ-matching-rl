#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:18:02 2017

@author: vitorhadad
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir

for f in listdir("results/"):
    
    if f.startswith("policy_") and f.endswith("pkl"):
        try:
            data = pickle.load(open("results/" + f, "rb"))
        except EOFError:
            print("Failed: ", f)
            continue
        fig, ax = plt.subplots()
        loss = pd.Series(data["training_losses"])
        ax = loss.rolling(len(loss)//100).mean().plot(ax = ax)
        ax.set_title(f.split(".")[0])