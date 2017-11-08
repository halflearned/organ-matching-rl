#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:54:33 2017

@author: vitorhadad
"""


import numpy as np
import pickle
from os import listdir 
from collections import defaultdict


def merge_data(*variables,
               path = "data/",
               horizon = 29,
               max_cycle_length = 2):
    files = listdir(path)
    Vs = defaultdict(list)
    for f in files:
        if f.endswith(".pkl"):            
            data = pickle.load(open(path + f, "rb"))
            for v in variables:
                Vs[v].extend([d[v] for d in data if 
                    d["horizon"] == horizon and 
                    d["max_cycle_length"] == max_cycle_length])
        
    
    return Vs




if __name__ == "__main__":
    
    
    path = "data/"
    
    data = merge_data("A", "X", "N", "G", "y", "opt_obj", 
                      path = path, 
                      horizon = 29, 
                      max_cycle_length = 2)
    
    #pickle.dump(data, open("training_data.pkl", "wb"))
    
    
    
    
    
    