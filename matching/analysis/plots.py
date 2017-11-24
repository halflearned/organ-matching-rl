#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:41:54 2017

@author: vitorhadad
"""


import pickle
#from matching.environment.data_utils import get_n_matched


files= [f for f in listdir("results/")
                if f.startswith("optn_")]

os = []
gs = []
for k, file in enumerate(files):
    print(k)
    
    data  = pickle.load(open("results/" + file, "rb"))
    env = data["env"]
    
    o = get_n_matched(data["opt_matched"], 0, env.time_length)
    g = get_n_matched(data["greedy_matched"], 0, env.time_length)
    
    os.append(o)
    gs.append(g)
    
#%%
gs = np.vstack(gs)
os = np.vstack(os)

#%%
ts = np.arange(1,gs.shape[1] + 1)


plt.plot(ts, os.mean(0))
plt.plot(ts, gs.mean(0))
plt.xlim((0,100))


#%%%




