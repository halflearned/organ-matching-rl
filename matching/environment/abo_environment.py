#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:05:52 2017

@author: vitorhadad
"""

import numpy as np
import pandas as pd
from collections import OrderedDict

from matching.environment.base_environment import BaseKidneyExchange, draw


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
        
        
    
    def draw_node_features(self, t_begin, t_end):
            
        duration = t_end - t_begin
        n_periods = np.random.poisson(self.entry_rate, size = duration)
        n = np.sum(n_periods)
        labels = ["entry", "death", "p_blood", "d_blood"]
        entries = np.repeat(np.arange(t_begin, t_end), n_periods)
        sojourns = np.random.geometric(self.death_rate, n)
        deaths = entries + sojourns
        p_blood = draw(self.blood_freq, n)
        d_blood = draw(self.blood_freq, n)
        return [dict(zip(labels, feats)) for feats in zip(entries,
                                                        deaths,
                                                        p_blood,
                                                        d_blood)]
        
      
        
    def draw_edges(self, source_nodes, target_nodes):
            
        np.random.seed(self.seed)
        source_nodes = np.array(source_nodes)
        target_nodes = np.array(target_nodes)
        
        source_entry = self.attr_to_numpy("entry", source_nodes)
        source_death = self.attr_to_numpy("death", source_nodes)
        source_don = self.attr_to_numpy("d_blood", source_nodes)
        
        target_entry = self.attr_to_numpy("entry", target_nodes)
        target_death = self.attr_to_numpy("death", target_nodes)
        target_pat = self.attr_to_numpy("p_blood", target_nodes)
        
        time_comp = (source_entry <= target_death.T) & (source_death >= target_entry.T)
        blood_comp = (source_don == target_pat.T) | (source_don == 0) | (target_pat.T == 3)
        not_same = source_nodes.reshape(-1,1) != target_nodes.reshape(1,-1)
        
        comp = time_comp & blood_comp & not_same
        
        s_idx, t_idx = np.argwhere(comp).T 
        
        return list(zip(source_nodes[s_idx], target_nodes[t_idx]))
        
        
        
        
        
        
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
    
        
    
    
        
        
        