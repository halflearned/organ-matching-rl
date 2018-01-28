#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 11:59:37 2017

@author: vitorhadad
"""

import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

tables = []

for env_type in ["optn", "abo"]:

    files = [f for f in glob("results/*.txt") 
            if  "optn" in f]
    
    logit_accs = {}
    count_accs = {}
    for f in files:
        name = f.split("/")[1]
        df = pd.read_csv(f, names = ["logit_loss",
                                     "count_loss",
                                     "tpr", "tnr",
                                     "logit_acc", "count_acc",
                                     "w"])
        logit_accs[name] = np.nanmedian(df["logit_acc"].values[-2000:])
        count_accs[name] = np.nanmedian(np.sqrt(df["count_loss"].values[-2000:]))
        
    logit_accs = pd.Series(logit_accs)
    count_accs = pd.Series(count_accs)
    tab = pd.concat([logit_accs, count_accs], 1)
    tab.columns = ["Label accuracy", "Count RMSE"]
    
    tables.append(tab)
        