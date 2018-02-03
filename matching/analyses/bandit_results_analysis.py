#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 19:58:30 2017

@author: vitorhadad
"""

import pandas as pd
import matplotlib.pyplot as plt
from os import system

#scp -P 22022 baisihad@sirius.bc.edu:/data/baisihad/matching/results/bandits_results3.txt results/

#with open("results/bandit_results3.txt", "r") as f:
#    old_lines = f.readlines()
#    new_lines = []
#    for line in old_lines:
#        line = line.replace('ABO', '"ABO')\
#                    .replace('RSU', '"RSU')\
#                    .replace('OPTN', '"OPTN')\
#                    .replace('None)', 'None)"')
#        new_lines.append(line)
#
#with open("results/bandit_results4.txt", "w") as f:
#    f.writelines(new_lines)
#    
#    
      
#df.loc[df["algorithm"]=="UCB1","algorithm"] = \
#    df.loc[df["algorithm"]=="UCB1","param"].apply(lambda c: "UCB1(c={})".format(c))
#df.loc[df["algorithm"] == "EXP3", "algorithm"] = \
#    df.loc[df["algorithm"]=="EXP3","param"].apply(lambda c: "EXP3(c={})".format(c))

#df["environment"] = df["environment"].fillna("ABO(5,0.1,1001,None)")

             
df = pd.read_csv("results/bandit_results4.txt")
    
df["max_time"] = df.groupby("seed")["time"]\
                   .transform(max)

df = df.query("max_time >= 1000")
                   

algorithm_perf = df.groupby(["environment", "algorithm", "thres"])\
                    [["rewards","greedy","opt"]]\
                    .agg(["mean", "std"])
                    
                    
algorithm_n = df.groupby(["environment", "algorithm", "thres"])\
                    ["seed"].nunique()      

by_time = df.groupby(["algorithm", "time", "thres"])\
                    [["rewards", "greedy", "opt"]]\
                    .mean()\
                    .unstack()

r= by_time["rewards"].T
g =by_time["greedy"].T

#ratio = (r/g).T
#
#fig, ax = plt.subplots(1, figsize=(10, 5))
#ax = ratio.T.rolling(100).mean().plot(ax = ax)
#ax.legend(bbox_to_anchor=(1.2, 1.05), fancybox=True, shadow=True)
#
#
#



