#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:05:52 2017

@author: vitorhadad
"""

import numpy as np
from matching.environment.base_environment import BaseKidneyExchange, draw
import pandas as pd
import networkx as nx
from collections import OrderedDict


class SaidmanKidneyExchange(BaseKidneyExchange):
    
   
    pra_freq = OrderedDict([("low", 0.7019), 
                            ("med", 0.2),
                            ("high", 0.0981)])
    
    cm_prob = OrderedDict([("low", 0.05),
                             ("med", 0.45),
                             ("high", 0.9)])
    
    blood_freq = OrderedDict([("o", 0.4814),
                  ("a", 0.3373),
                  ("b", 0.1428),
                  ("ab", 0.0385)])
    
    gender_freq = OrderedDict([("male", 0.5910),
                   ("female", 0.4090)])
    
    spouse_freq = OrderedDict([("yes", 0.4897),
                                 ("no", 0.5103)])
    
    spouse_cm_prob_scaling = 0.75
    
    

    def __init__(self, 
                 entry_rate,
                 death_rate,
                 time_length = 400,
                 seed = None,
                 populate=True):
        
        super(SaidmanKidneyExchange, self)\
              .__init__(entry_rate=entry_rate,
                        death_rate=death_rate,
                        time_length=time_length,
                        seed=seed,
                        populate=populate)
        
        
    
    def populate(self, t_begin = None, t_end = None, seed = None):
        
        t_begin = t_begin or 0
        t_end = t_end or self.time_length
        seed = seed or self.seed
        
        np.random.seed(seed)
    
        self.erase_from(t_begin)
        i_cur = self.number_of_nodes()
        
        duration = t_end - t_begin
        n_today = np.random.poisson(self.entry_rate, size = duration)
        n = np.sum(n_today)
        
        entries = np.repeat(np.arange(t_begin, t_end), n_today)
        sojourns = np.random.geometric(self.death_rate, n)
        deaths = entries + sojourns
        p_blood = draw(self.blood_freq, n)
        d_blood = draw(self.blood_freq, n)
        is_female = draw(self.gender_freq, n)
        pra = np.random.choice(list(self.cm_prob.values()),
                               p = list(self.pra_freq.values()),
                               size = n)
        
        for i in range(n):
            self.add_node(i + i_cur,
                          entry = entries[i],
                          death = deaths[i],
                          p_blood = p_blood[i],
                          d_blood = d_blood[i],
                          is_female = is_female[i],
                          pra = pra[i])
            
        
        pra_compatible = np.random.uniform(size = (n, n)) < pra 
        pra_compatible[np.arange(n), np.arange(n)] = False
        contemporaneous = self.is_contemporaneous(entries, deaths)  
        blood_compatible = self.is_blood_compatible(d_blood, p_blood)

        compatible = np.argwhere(pra_compatible & \
                                 contemporaneous & \
                                 blood_compatible)
        
        self.add_edges_from(i_cur + compatible, weight = 1)
        
        
    @staticmethod
    def is_contemporaneous(entries, deaths):
        return (entries.reshape(-1, 1) <= deaths.flatten()) & \
                (deaths.reshape(-1, 1) >= entries.flatten())
        

    @staticmethod
    def is_blood_compatible(d_blood, p_blood):
        d_blood = np.array(d_blood).reshape(-1, 1)
        p_blood = np.array(p_blood).reshape(1, -1)
        return (d_blood == p_blood) | (p_blood == 3) | (d_blood == 0)

        
        
    def X(self, t, dtype = "numpy"):
        
        nodelist = self.get_living(t, indices_only = False)
        n = len(nodelist)
        Xs = np.zeros((n, 10))
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
            Xs[i, 8] = d["is_female"]
            Xs[i, 9] = d["pra"]
            indices.append(n)
            
        if dtype == "pandas":
            return pd.DataFrame(index = indices,
                         data= Xs,
                         columns = ["pO","pA","pAB",
                                       "dO","dA","dB",
                                       "waiting_time",
                                       "time_to_death",
                                       "is_female",
                                       "pra"])
        elif dtype == "numpy":
            return Xs
        else:
            raise ValueError("Invalid dtype")
        
        

    
#%%
if __name__ == "__main__":
    
    
    env = SaidmanKidneyExchange(entry_rate  = 5,
                                death_rate  = 0.1,
                                time_length = 20)

    A, X = env.A(3), env.X(3)
    
        
#    g = env.generate_cycles(2)
#    cycles = list(g)
#    
#    
#    g = env.generate_cycles(3)
#    cycles = list(g)
    
    
    g = env.generate_cycles(2, env.get_living(0))
    cycles = list(g)
        
        
        