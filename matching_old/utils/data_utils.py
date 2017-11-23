#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:50:59 2017



@author: vitorhadad
"""

import numpy as np
from tqdm import trange
import pickle
from time import time
from scipy.stats import geom
import networkx as nx
import pandas as pd
from os import listdir
from collections import defaultdict

from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.solver.kidney_solver import KidneySolver


def clock_seed():
    return int(str(int(time()*1e8))[10:])  





def get_rewards(solution, t, h):
    return sum([len(match) for period, match in solution["matched"].items()
                if period >= t and period < t + h])

    

def get_additional_regressors(env, t):
    
    f = lambda d: list(d.values())
    
    nodes = env.get_living(t)
    subg = nx.subgraph(env, nodes)
    
    graph_properties = pd.DataFrame({
        "avg_node_connectivity": nx.average_node_connectivity(subg),
        "density": nx.density(subg),
        "number_of_nodes": [subg.number_of_nodes()]*subg.number_of_nodes(),
        "number_of_edges": [subg.number_of_edges()]*subg.number_of_nodes()
        })

    node_properties = {}
    try:
        node_properties["betweenness_centrality"] = f(nx.betweenness_centrality(subg))
    except:
        node_properties["betweenness_centrality"] = [0]*subg.number_of_nodes()
    
    try:
        node_properties["in_degree_centrality"] = f(nx.in_degree_centrality(subg))
    except:
        node_properties["in_degree_centrality"] = [0]*subg.number_of_nodes()
        
    try:
        node_properties["out_degree_centrality"] = f(nx.out_degree_centrality(subg))
    except:
        node_properties["out_degree_centrality"] = [0]*subg.number_of_nodes()
      
    try:
        node_properties["harmonic_centrality"] = f(nx.harmonic_centrality(subg))
    except:
        node_properties["harmonic_centrality"] = [0]*subg.number_of_nodes()
    
    try:
        node_properties["closeness_centrality"] = f(nx.closeness_centrality(subg))
    except:
        node_properties["closeness_centrality"] = [0]*subg.number_of_nodes()
    
    
    node_properties.update({
       "core_number": f(nx.core_number(subg)),
       "pagerank": f(nx.pagerank(subg)),
       "in_edges":  [len(subg.in_edges(v)) for v in subg.nodes()],
       "out_edges": [len(subg.out_edges(v)) for v in subg.nodes()],
       "average_neighbor_degree": f(nx.average_neighbor_degree(subg))
    })
    
    node_properties = pd.DataFrame(node_properties)
    
    
    return graph_properties, node_properties
        




    
def merge_data(*variables,
               path = "data/"):
    files = listdir(path)
    data = defaultdict(list)
    for f in files:
        if f.startswith("data_with") and f.endswith(".pkl"): 
            try:
                pkl = pickle.load(open(path + f, "rb"))
                for dt in pkl:
                    for v in variables:
                        data[v].append(dt[v])
                        
            except EOFError:
                    print("Could not open: ", f);
                
    return data

    

def get_n_matched(algo):
    n_matched = np.zeros(max(algo["matched"]) + 1)
    for t, m in algo["matched"].items():
        n_matched[t] = len(m)
    return n_matched




def prepare_mdp(n,
                entry_rate = 5,
                death_rate = .1,
                max_cycle_length = 2,
                time_length = 200):
        
    seed = clock_seed()
    solver = KidneySolver(max_cycle_length = max_cycle_length,
                          burn_in = 0)  
    
    data = []
    for i_iter in trange(n, desc = "Preparing data"):
    
        env = SaidmanKidneyExchange(entry_rate  = entry_rate,
                                    death_rate  = death_rate,
                                    time_length = time_length,
                                    seed = seed)

        opt = solver.optimal(env)
        greedy = solver.greedy(env)
        
        for t in range(env.time_length):
            
            m = opt["matched"].get(t, [])
            y = np.zeros(len(env.get_living(t)))
            y[env.reindex_to_period(m, t)] = 1
            X = env.X(t)
            A = env.A(t, "sparse")
            G, N = get_additional_regressors(env, t)
            env.removed_container[t].update(m)
            data.append({"X": X,
                         "A": A,
                         "G": G,
                         "N": N,
                         "t": t,
                         "y": y,
                         "opt_obj": opt["obj"],
                         "greedy_obj": greedy["obj"],
                         "opt_n_matched": get_n_matched(opt),
                         "greedy_n_matched": get_n_matched(greedy),
                         "max_cycle_length": max_cycle_length,
                         "entry_rate": entry_rate,
                         "death_rate": death_rate,
                         "time_length": time_length,
                         "seed": seed})
        
    return data



def oversample(X, y, frac_ones = 0.5):
    y_is_one = y.astype(bool)
    
    n_zeros = np.sum(~y_is_one).astype(int)
    X_zeros = X[~y_is_one]
    
    n_ones = np.sum(y_is_one).astype(int)
    X_ones = X[y_is_one]
    idx = np.random.randint(n_ones, size = n_zeros)
    X_ones_resampled = X_ones[idx]
    
    X_resampled = np.vstack([X_ones_resampled, X_zeros])
    y_resampled = np.hstack([np.ones(n_zeros), np.zeros(n_zeros)])
    sh = np.random.permutation(len(y_resampled))
    return X_resampled[sh], y_resampled[sh]
    


def undersample(X, y, frac_ones = 0.5):
    y_is_one = y.astype(bool)
    
    # Separate regressors associated with y=1
    n_ones = np.sum(y_is_one).astype(int)
    X_ones = X[y_is_one]
    
    # Now collect regressors associated with y=0
    n_zeros = np.sum(~y_is_one).astype(int)
    X_zeros = X[~y_is_one]
    # And resampled the same number of y=1's
    idx = np.random.randint(n_zeros, size = n_ones)
    X_zeros_resampled = X_zeros[idx]
    
    X_resampled = np.vstack([X_ones, X_zeros_resampled])
    y_resampled = np.hstack([np.ones(n_ones), np.zeros(n_ones)])
    sh = np.random.permutation(len(y_resampled))
    return X_resampled[sh], y_resampled[sh]

#%%
if __name__ == "__main__":
    
    while True:
        #path = "/Users/vitorhadad/Documents/kidney/matching/data/"
        try:
            path = "data/"
            name = str(np.random.randint(1e8))
            data = prepare_mdp(1, time_length = 200)
            pickle.dump(data, open(path + "data_with_value_{}.pkl".format(name), "wb"))
        except:
            pass
