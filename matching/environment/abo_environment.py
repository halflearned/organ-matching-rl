#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:05:52 2017

@author: vitorhadad
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from itertools import product

from matching.environment.base_environment import BaseKidneyExchange, draw


class ABOKidneyExchange(BaseKidneyExchange):
    
    # Computed using pc = 0.11
    blood_types = np.vstack([(0, 0), (0, 1), (0, 2), (0, 3),
                           (1, 0), (1, 1), (1, 2), (1, 3),
                           (2, 0), (2, 1), (2, 2), (2, 3),
                           (3, 0), (3, 1), (3, 2), (3, 3)])
    
    blood_prob = np.array([0.05868620827131898, 0.37381232862325586,
                  0.1582579321891519, 0.04266757975687919,
                  0.04111935614855814, 0.02881088248630798,
                  0.11088575099169286, 0.029895668159525032,
                  0.01740837254080671, 0.11088575099169286,
                  0.005163929370226835, 0.01265668963290891,
                  0.004693433773256711, 0.0032885234975477537,
                  0.0013922358596199801, 0.0003753577072504848])
    
    
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
        
        
    def draw_node_features(self, t_begin, t_end):
        
        if t_begin == 0:
            np.random.seed(self.seed)
        
        duration = t_end - t_begin
        n_periods = np.random.poisson(self.entry_rate, size = duration)
            
        n = np.sum(n_periods)
        entries = np.repeat(np.arange(t_begin, t_end), n_periods).reshape(-1, 1)
        
        sojourns = np.random.geometric(self.death_rate, size=(n,1)) - 1
        deaths = entries + sojourns
        
        idx = np.random.choice(len(self.blood_types),
                               p = self.blood_prob,
                               size = n)
        
        blood = self.blood_types[idx]
       
        data = np.hstack([entries, deaths, blood])
        
        colnames = ["entry", "death", "p_blood", "d_blood"]
        results = []
        for row in data:
            results.append(dict(zip(colnames, row)))
        
        return results
        
        

        
    def draw_edges(self, source_nodes, target_nodes):
            
        np.random.seed(self.seed)
        source_nodes = np.array(source_nodes)
        target_nodes = np.array(target_nodes)
        
        source_entry = self.attr("entry", nodes = source_nodes)
        source_death = self.attr("death", nodes = source_nodes)
        source_don = self.attr("d_blood", nodes = source_nodes)
        
        target_entry = self.attr("entry", nodes = target_nodes)
        target_death = self.attr("death", nodes = target_nodes)
        target_pat = self.attr("p_blood", nodes = target_nodes)
        
        time_comp = (source_entry <= target_death.T) & (source_death >= target_entry.T)
        blood_comp = (source_don == target_pat.T) | (source_don == 0) | (target_pat.T == 3)
        not_same = source_nodes.reshape(-1,1) != target_nodes.reshape(1,-1)
        
        comp = time_comp & blood_comp & not_same
        
        s_idx, t_idx = np.argwhere(comp).T 
        
        return list(zip(source_nodes[s_idx], target_nodes[t_idx]))
        
        
        
        
        
        
    def X(self, t, graph_attributes = True, dtype = "numpy"):
        
        nodelist = self.get_living(t, indices_only = False)
        n = len(nodelist)
        Xs = np.zeros((n, 8 + 2*graph_attributes))
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
            if graph_attributes:
                Xs[i, 8] = self.entry_rate
                Xs[i, 9] = self.death_rate
            indices.append(n)
            
        if dtype == "numpy":
            return Xs
        
        elif dtype == "pandas":
            columns = ["pO","pA","pAB",
                       "dO","dA","dB",
                       "waiting_time",
                       "time_to_death"]
            if graph_attributes:
                columns += ["entry_rate", "death_rate"]
            return pd.DataFrame(index = indices,
                                data= Xs,
                                columns = columns)
        else:
            raise "Unknown dtype"
        
        

    
#%%
if __name__ == "__main__":
    
    
    env = ABOKidneyExchange(entry_rate  = 5,
                            death_rate  = 0.1,
                            time_length = 20,
                            seed = 12345)

    A, X = env.A(3), env.X(3)
    
        
    
    
        
        
        