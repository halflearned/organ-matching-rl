#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:50:59 2017

Creates tuples of the form

(X(t), A(t), greedy_obj[t,t+h], opt_obj[t,t+h])

where [[method]]_obj[t,t+h] refers to the cardinality of 
matched pairs by [[method]] between periods t and h.

The horizons h are typically .9, .95, .99 quantiles 
of the death rate distribution.


@author: vitorhadad
"""

from kidney_solver import KidneySolver
import numpy as np
from saidman_environment import SaidmanKidneyExchange
from tqdm import trange
import pickle
from time import time
from scipy.stats import geom
import networkx as nx
import pandas as pd

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

    node_properties = pd.DataFrame({
       "degree_centrality": f(nx.degree_centrality(subg)),
       "betweenness_centrality": f(nx.betweenness_centrality(subg)),
       "in_degree_centrality": f(nx.in_degree_centrality(subg)),
       "out_degree_centrality": f(nx.out_degree_centrality(subg)),
       "core_number": f(nx.core_number(subg)),
       "pagerank": f(nx.pagerank(subg)),
       "closeness_centrality": f(nx.closeness_centrality(subg)),
       "in_edges":  [len(subg.in_edges(v)) for v in subg.nodes()],
       "out_edges": [len(subg.out_edges(v)) for v in subg.nodes()],
       "harmonic_centrality": f(nx.harmonic_centrality(subg)),
       "average_neighbor_degree": f(nx.average_neighbor_degree(subg))
    })
   
    return graph_properties, node_properties
        
    
    

def prepare_mdp(n,
                entry_rate = 5,
                death_rate = .1,
                max_cycle_length = 2,
                time_length = 100,
                horizons = None):
    
    if horizons is None:
        horizons = geom(death_rate).ppf([.90, .95, .99]).astype(int)
    
    max_horizon = horizons[-1]
    
    seed = clock_seed()
    solver = KidneySolver(max_cycle_length = max_cycle_length,
                          burn_in = 0)  
    
    data = []
    
    for i_iter in trange(n, desc = "Preparing data"):
    
        env = SaidmanKidneyExchange(entry_rate  = entry_rate,
                                    death_rate  = death_rate,
                                    time_length = time_length + max_horizon,
                                    seed = seed)

        opt = solver.optimal(env)
        
        
        for t in range(env.time_length - max_horizon):
            
            m = opt["matched"].get(t, [])
            
    
            y = np.zeros(len(env.get_living(t)))
            y[env.reindex_to_period(m, t)] = 1
            
            env.removed_container[t].update(m)
    
            greedy = solver.greedy(env, 
                                   horizon = 0,
                                   t_begin = t,
                                   t_end = t + max_horizon)
            
            G, N = get_additional_regressors(env, t)
            
            for h in horizons:

                data.append({"X": env.X(t),
                             "A": env.A(t, "sparse"),
                             "G": G,
                             "N": N,
                             "t": t,
                             "y": y,
                             "horizon": h,
                             "opt_obj": get_rewards(opt, t, h),
                             "greedy_obj": get_rewards(greedy, t, h),
                             "max_cycle_length": max_cycle_length,
                             "entry_rate": entry_rate,
                             "death_rate": death_rate,
                             "time_length": time_length,
                             "seed": seed})
    
    return data



if __name__ == "__main__":
    
    name = str(np.random.randint(1e8))
    data = prepare_mdp(1)
    #pickle.dump(data, open("value_function_data_{}.pkl".format(name), "wb"))

