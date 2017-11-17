#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:17:27 2017

@author: vitorhadad
"""

import pandas as pd
import numpy as np
import pickle

from matching.environment.base_environment import BaseKidneyExchange

pat = pickle.load(open("matching/optn_data/patient_info.pkl", "rb"))
don = pickle.load(open("matching/optn_data/donor_info.pkl", "rb"))


class OPTNKidneyExchange(BaseKidneyExchange):
    
    def __init__(self, 
             entry_rate,
             death_rate,
             time_length,
             seed=None,
             populate=True):
    
        super(self.__class__, self)\
                  .__init__(entry_rate=entry_rate,
                    death_rate=death_rate,
                    time_length=time_length,
                    seed=seed,
                    populate=False)
          
        self.patient = pat
        self.donor = don
        if populate: self.populate(seed)
        

    def __class___(self):
        return OPTNKidneyExchange
    
    
    def sdadasdad(self):
        pass


    def draw_node_features(self, t_begin, t_end):
        np.random.seed(self.seed)
        
        duration = t_end - t_begin
        n_periods = np.random.poisson(self.entry_rate, size = duration)
        n_periods[0] = self.entry_rate # Make sure someone exists at t=0
        
        n = np.sum(n_periods)
        entries = np.repeat(np.arange(t_begin, t_end), n_periods)
        
        sojourns = np.random.geometric(self.death_rate, n)
        deaths = entries + sojourns
        
        pat_rnd_idx = np.random.randint(len(self.patient), size = n)
        this_pat = self.patient.iloc[pat_rnd_idx].reset_index(drop=True)
        
        don_rnd_idx = np.random.randint(len(self.donor), size = n)
        this_don = self.donor.iloc[don_rnd_idx].reset_index(drop=True)
        
        out = pd.concat([this_pat, this_don], axis = 1)
        
        out["entry"] = entries
        out["death"] = deaths
        cols = out.columns
        return out.apply(lambda x: dict(zip(cols, x.values)), axis = 1).tolist()

        


    def draw_edges(self, source_nodes, target_nodes):
        edges = []
        for s in source_nodes:
            for t in target_nodes:
                
                if s == t:
                    continue
                
                s_data = self.node[s]
                t_data = self.node[t]
                
                time_comp = s_data["entry"] <= t_data["death"] and \
                            s_data["death"] >= t_data["entry"]
                if not time_comp:
                    continue
                
                blood_comp = s_data["patient_blood"] == "AB" or \
                             t_data["donor_blood"] == "O" or \
                             s_data["patient_blood"] == t_data["donor_blood"]
                if not blood_comp:
                    continue
                
                histocomp = s_data["donor_A"].isdisjoint(t_data["notA"]) and \
                            s_data["donor_B"].isdisjoint(t_data["notB"]) and \
                            s_data["donor_C"].isdisjoint(t_data["notC"]) and \
                            s_data["donor_DR"].isdisjoint(t_data["notDR"])
                if not histocomp:
                    continue
                
                edges.append((s, t))
                
        return edges


            
    def X(self, t, dtype = "numpy"):
        nodelist = self.get_living(t, indices_only = False)
        n = len(nodelist)
        Xs = np.zeros((n, 10))
        indices = []
        for i, (n, d) in enumerate(nodelist):
            Xs[i, 0] =  d["patient_blood"] == "O"
            Xs[i, 1] =  d["patient_blood"] == "A"
            Xs[i, 2] =  d["patient_blood"] == "B"
            Xs[i, 3] =  d["donor_blood"] == "O"
            Xs[i, 4] =  d["donor_blood"] == "A"
            Xs[i, 5] =  d["donor_blood"] == "B"
            Xs[i, 6] = t - d["entry"] 
            Xs[i, 7] = d["death"] - t
            indices.append(n)
            
        if dtype == "pandas":
            return pd.DataFrame(index = indices,
                         data= Xs,
                         columns = ["pO","pA","pB",
                                    "dO","dA","dB",
                                    "waiting_time",
                                    "time_to_death"])
        elif dtype == "numpy":
            return Xs
        else:
            raise ValueError("Invalid dtype")

    
    
#%%    
if __name__ == "__main__":
    
    from matching.solver.kidney_solver import KidneySolver
    import matplotlib.pyplot as plt
   
    T = 10
    env = OPTNKidneyExchange(5, .1, T, populate=True)
    solver = KidneySolver(2)
    opt = solver.optimal(env)
    greedy = solver.greedy(env)
    print("OPT/GREEDY Ratio:", opt["obj"]/greedy["obj"])
    #%%
    mopt = np.zeros(T)
    gopt = np.zeros(T)
    for t in range(T):
        mopt[t] = len(opt["matched"][t])
        gopt[t] = len(greedy["matched"][t])
    moptc = np.cumsum(mopt) / np.arange(1, T+1)
    goptc = np.cumsum(gopt) / np.arange(1, T+1)
    
    #%%
    plt.plot(moptc)
    plt.plot(goptc)
    plt.legend(["OPT", "Greedy"])
    plt.title("Average matches per period (entry_rate:{}, death_rate:{})"\
              .format(env.entry_rate, env.death_rate))
    
    