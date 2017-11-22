#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:05:52 2017

@author: vitorhadad
"""

import numpy as np
from base_environment import BaseKidneyExchange, draw
import pandas as pd
import networkx as nx
from collections import OrderedDict


class ABOKidneyExchange(BaseKidneyExchange):
    
   
    blood_freq = OrderedDict([("o", 0.4814),
                              ("a", 0.3373),
                              ("b", 0.1428),
                              ("ab", 0.0385)])

    def __init__(self, 
                 entry_rate,
                 death_rate,
                 time_length = 400,
                 seed = None,
                 populate = True):
        
        
        super(ABOKidneyExchange, self)\
              .__init__(entry_rate=entry_rate,
                        death_rate=death_rate,
                        time_length=time_length,
                        seed=seed,
                        populate=populate)
        
        
    
    def populate(self):
        
        
        n_today = np.random.poisson(self.entry_rate, size=self.time_length)
        n = np.sum(n_today)
        entries = np.repeat(np.arange(self.time_length), n_today)
        sojourns = np.random.geometric(self.death_rate, n)
        deaths = entries + sojourns
        p_blood = draw(self.blood_freq, n)
        d_blood = draw(self.blood_freq, n)
        
        
        for i in range(n):
            self.add_node(i, 
                       entry = entries[i],
                       death = deaths[i],
                       p_blood = p_blood[i],
                       d_blood = d_blood[i])
                      
            
        contemporaneous = (entries.reshape(-1, 1) <= deaths) & \
                          (deaths.reshape(-1, 1) >= entries)

        blood_compatible = (p_blood.reshape(-1,1) == d_blood) | \
                           (p_blood.reshape(-1,1) == 3) | \
                           (d_blood.reshape(-1,1) == 0)
                           
        compatible = np.argwhere(contemporaneous & \
                                 blood_compatible)
        
        self.add_edges_from(compatible, weight = 1)
        

        
        
    def X(self, t):
        
        nodelist = self.get_living(t, indices_only = False)
        n = len(nodelist)
        Xs = np.zeros((n, 8))
        indices = []
        for i, (n, d) in enumerate(nodelist):
            Xs[i, 0] =  d["p_blood"] == 0
            Xs[i, 1] =  d["p_blood"] == 1
            Xs[i, 2] =  d["p_blood"] == 2
            Xs[i, 3] =  d["d_blood"] == 0
            Xs[i, 4] =  d["d_blood"] == 1
            Xs[i, 5] =  d["d_blood"] == 2
            Xs[i, 6] = t - d["entry"] 
            Xs[i, 7] = d["death"] - t
            indices.append(n)
            
        return pd.DataFrame(index = indices,
                            data= Xs,
                            columns = ["pO","pA","pAB",
                                       "dO","dA","dB",
                                       "waiting_time",
                                       "time_to_death"])
        
        

    
#%%
if __name__ == "__main__":
    
    
    env = ABOKidneyExchange(entry_rate  = 5,
                            death_rate  = 0.1,
                            time_length = 20,
                            seed = 12345)

    A, X = env.A(3), env.X(3)
    
        
    
    
        
        
        