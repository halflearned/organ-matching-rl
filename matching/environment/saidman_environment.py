

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:05:52 2017

@author: vitorhadad
"""

import numpy as np
from matching.environment.base_environment import BaseKidneyExchange, draw
import pandas as pd
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
        
        super(self.__class__, self)\
              .__init__(entry_rate=entry_rate,
                        death_rate=death_rate,
                        time_length=time_length,
                        seed=seed,
                        populate=populate)
        

        
        
    
    def draw_node_features(self, t_begin, t_end):
        duration = t_end - t_begin
        n_periods = np.random.poisson(self.entry_rate, size = duration)
        n = np.sum(n_periods)
        labels = ["entry", "death", "p_blood",
                  "d_blood", "is_female", "pra"]
        entries = np.repeat(np.arange(t_begin, t_end), n_periods)
        sojourns = np.random.geometric(self.death_rate, n)
        deaths = entries + sojourns
        p_blood = draw(self.blood_freq, n)
        d_blood = draw(self.blood_freq, n)
        is_female = draw(self.gender_freq, n)
        pra = np.random.choice(list(self.cm_prob.values()),
                               p = list(self.pra_freq.values()),
                               size = n)
        return [dict(zip(labels, feats)) for feats in zip(entries,
                                                        deaths,
                                                        p_blood,
                                                        d_blood,
                                                        is_female,
                                                        pra)]
        
      
        
    def draw_edges(self, source_nodes, target_nodes):
        
        source_nodes = np.array(source_nodes)
        target_nodes = np.array(target_nodes)
        
        ns = len(source_nodes)
        nt = len(target_nodes)
        
        source_entry = self.attr_to_numpy("entry", source_nodes)
        source_death = self.attr_to_numpy("death", source_nodes)
        source_don = self.attr_to_numpy("d_blood", source_nodes)
        source_pra = self.attr_to_numpy("pra", source_nodes)
        
        target_entry = self.attr_to_numpy("entry", target_nodes)
        target_death = self.attr_to_numpy("death", target_nodes)
        target_pat = self.attr_to_numpy("p_blood", target_nodes)
        
        hist_comp = np.random.uniform(size = (ns, nt)) > source_pra
        time_comp = (source_entry <= target_death.T) & (source_death >= target_entry.T)
        blood_comp = (source_don == target_pat.T) | (source_don == 0) | (target_pat.T == 3)
        not_same = source_nodes.reshape(-1,1) != target_nodes.reshape(1,-1)
        
        comp = hist_comp & time_comp & blood_comp & not_same
        
        s_idx, t_idx = np.argwhere(comp).T 
        
        return list(zip(source_nodes[s_idx], target_nodes[t_idx]))
        
        
        
        
#     Use this for testing later    
#    def draw_edges2(self, source_nodes, target_nodes):
#        edges = []
#        for s in source_nodes:
#            for t in target_nodes:
#                
#                if s == t:
#                    continue
#                
#                s_data = self.node[s]
#                hist_comp = np.random.uniform() > s_data["pra"] 
#                if not hist_comp:
#                    continue
#                
#                t_data = self.node[t]
#                time_comp = s_data["entry"] <= t_data["death"] and \
#                            s_data["death"] >= t_data["entry"]
#                if not time_comp:
#                    continue
#                
#                blood_comp = s_data["d_blood"] == 0 or \
#                             t_data["p_blood"] == 3 or \
#                             s_data["d_blood"] == t_data["p_blood"]
#                if not blood_comp:
#                    continue
#                
#                edges.append((s, t))
#                
#        return edges
            
        
        

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
                         columns = ["pO","pA","pB",
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
                                time_length = 100)

    A, X = env.A(3), env.X(3)

        
        
        