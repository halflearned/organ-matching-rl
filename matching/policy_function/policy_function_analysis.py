#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:26:55 2017

@author: vitorhadad
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir

path= "policy_function_results/"

tabs = []
for f in listdir(path):
    
    if f.startswith("policy_function_traditional_"):
        
        name = path + f
        try:
            r = pickle.load(open(name, "rb"))
        except EOFError:
            print("Error when loading", f)
        
        this = np.array(r["this"])
        g = np.array(r["greedy"])
        
        thresholds = np.linspace(0.05, 0.95, 19)
        
        ms = []
        for i in range(19):
            ms.append(np.mean(this[i::19] - g[i::19]))
        tab = pd.Series(index = thresholds, data= ms)
        tab.name = str(r["algo"])
        tabs.append(tab)

tabs = pd.concat(tabs, axis = 1)

#
#
#for i in range(10, 11, 1):
#    plt.plot(this[i::19] - g[i::19], label = str(thresholds[i]))
#plt.title(r["algo"])
#plt.legend()