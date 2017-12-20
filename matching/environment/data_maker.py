#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:46:23 2017

@author: vitorhadad
"""

import pickle
import numpy as np
from sys import argv
import re
from tqdm import trange

import matching.utils.data_utils as utils

    
def get_regressors(file, times):
    As = []
    Xs = []
    GNs = []
    Ys = []
    file["env"].removed_container.clear()
    for t in trange(file["env"].time_length):        
        if t in times:
            liv = file["env"].get_living(t)
            A = file["env"].A(t, "sparse")
            X = file["env"].X(t, graph_attributes = True)
            G, N = utils.get_additional_regressors(file["env"], t)
            
            Y = np.zeros_like(liv).reshape(-1, 1)
            Y[np.isin(liv, list(file["opt"]["matched"][t]))] = 1
            
            As.append(A)
            Xs.append(X)
            GNs.append(np.hstack([G,N]))
            Ys.append(Y)
            
        file["env"].removed_container[t].update(file["opt"]["matched"][t])
    
    file["env"].removed_container.clear()
    return As, Xs, GNs, Ys
    

path = "results/"


filename = argv[1]
envtype = re.search("^[a-z]+", filename).group()

try:
    file = pickle.load(open(path + filename, "rb"))
except Exception as e:
    print("Exception:", e)
    

random_times = np.sort(np.random.randint(file["env"].time_length, size = 200))
   
As, Xs, GNs, Ys = get_regressors(file, random_times)  

pickle.dump(obj = {"A": As, "G": GNs, "X": Xs, "Y": Ys},
            file = open("data/data_{}_{}.pkl"\
                        .format(envtype), "wb"))

        
        
        
        

