#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:50:59 2017
z


@author: vitorhadad
"""

import numpy as np
from tqdm import trange
from time import time
import networkx as nx
import pandas as pd
from os import listdir
from random import choice
from re import findall
from torch.autograd import Variable

from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.solver.kidney_solver import KidneySolver

#%%


def softmax(x , T = 1):
    e_x = np.exp((x - np.max(x))/T)
    return e_x / e_x.sum()


def confusion(y_pred, y_true, ls): 
    
    if isinstance(y_pred, Variable):
        y_pred = y_pred.data.numpy()
    
    y_true = np.concatenate([y_true[k,:l,0] for k,l in enumerate(ls)])
    y_pred = np.concatenate([y_pred[k,:l,1] > y_pred[k,:l,0] for k,l in enumerate(ls)])

    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fn = np.sum(y_true & ~y_pred)
    
    return tp,tn,fp,fn



def open_file(env_type = "optn", 
              mcl = 2, 
              open_A = True,
              open_GN = True):
    
    prefix = "{}_mcl{}".format(env_type, mcl)
    f = choice([f for f in listdir("data/") if prefix in f])
    number = findall("[0-9]{4,}", f)[0]
    
    X  = np.load("data/{}_X_{}.npy".format(prefix, number))
    Y  = np.load("data/{}_Y_{}.npy".format(prefix, number))
    if not open_A and not open_GN:
        return X, Y
    if open_A and not open_GN:
        A  = np.load("data/{}_A_{}.npy".format(prefix, number))
        return A, X, Y
    elif open_GN and not open_A:
        GN  = np.load("data/{}_GN_{}.npy".format(prefix, number))
        return X, Y, GN
    elif open_GN and open_A:
        A  = np.load("data/{}_A_{}.npy".format(prefix, number))
        GN  = np.load("data/{}_GN_{}.npy".format(prefix, number))
        return A, X, GN, Y
    

def get_size(env, m):
    T = env.time_length
    matched = set()
    size = []
    for t in range(T):
        living = set(env.get_living(t))
        matched.update(m[t])
        size.append(len(living - matched))  
    return size



def cumavg(x):
    return np.cumsum(x)/np.arange(1, len(x)+1)


def clock_seed():
    return int(str(int(time()*1e8))[10:])  


def disc_mean(xs, gamma = 0.97):
    return np.mean([gamma**i * r for i,r in enumerate(xs)])


def balancing_weights(XX, y):
    yy = y.flatten()
    n1 = np.sum(yy)
    n0 = len(yy) - n1
    p = np.zeros(int(n0 + n1))
    p[yy == 0] = 1/n0
    p[yy == 1] = 1/n1
    p /= p.sum()
    return p


    

def get_rewards(solution, t, h):
    return sum([len(match) for period, match in solution["matched"].items()
                if period >= t and period < t + h])

    

def flatten_matched(m, t_begin = 0, t_end = None):
    if t_end is None:
        t_end = np.inf
    matched = []
    for t,x in m.items():
        if t >= t_begin and t <= t_end:
            matched.extend(x)
    return set(matched)
    

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






def get_dead(env, matched, t_begin = None, t_end = None):
    
    if t_begin is None:
        t_begin = 0

    if t_end is None:
        t_end = env.time_length-1
        
    would_be_dead = {n for n, d in env.nodes.data() 
                    if d["death"] >= t_begin and \
                       d["death"] <= t_end}
    
    dead = would_be_dead.difference(matched)
    
    return dead




def pad_and_stack(As, Xs, GNs, Ys):
    n = max(x.shape[0] for x in Xs)
    A_pad = []
    X_pad = []
    GN_pad = []
    Y_pad = []
    for A, X, GN, Y in zip(As, Xs, GNs, Ys):
        r = n - X.shape[0]
        A_pad.append(np.pad(A, ((0,r),(0,r)), mode = "constant"))
        X_pad.append(np.pad(X, ((0,r),(0,0)), mode = "constant"))
        GN_pad.append(np.pad(X, ((0,r),(0,0)), mode = "constant"))
        Y_pad.append(np.pad(Y, ((0,r),(0,0)), mode = "constant"))
    return np.stack(A_pad),np.stack(X_pad),\
                np.stack(GN_pad), np.stack(Y_pad)
            
        

def get_regressors(file, size):
    As = []
    Xs = []
    GNs = []
    Ys = []
    file["env"].removed_container.clear()
    random_times = np.sort(np.random.randint(file["env"].time_length, size = size))
    for t in trange(file["env"].time_length):
        if t in random_times:
            liv = file["env"].get_living(t)
            if len(liv) == 0:
                continue
            A = file["env"].A(t, dtype = "numpy")
            X = file["env"].X(t, dtype = "numpy", graph_attributes = True)
            G, N = get_additional_regressors(file["env"], t)
            
            Y = np.zeros_like(liv).reshape(-1, 1)
            Y[np.isin(liv, list(file["opt"]["matched"][t]))] = 1
            
            As.append(A)
            Xs.append(X)
            GNs.append(np.hstack([G,N]))
            Ys.append(Y)

        file["env"].removed_container[t].update(file["opt"]["matched"][t])

    file["env"].removed_container.clear()
    return As, Xs, GNs, Ys




    
def get_deaths(env, solution, t_begin = None, t_end = None):

    if t_begin is None:
        t_begin = 0
    
    if t_end is None:
        t_end = env.time_length
        
    R = solution["matched_pairs"]

    deaths = np.zeros(t_end - t_begin)
    
    for t in range(env.time_length):
        dead = {n for n,d in env.nodes.data() 
                    if d["death"] == t
                    and n not in R}
        deaths[t] = len(dead)
        
    return deaths





def get_n_matched(matched, t_begin, t_end):

    n_matched = np.zeros(t_end - t_begin)
    
    for t in range(t_begin, t_end):
        n_matched[t - t_begin] = len(matched[t])
        
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
def stata_to_csv(filepath, outfile, chunksize = 10000):
    
    file = pd.read_stata(filepath, 
                         iterator = True,
                         chunksize = chunksize)
    
    for k, b in enumerate(file):
        if k % 10 == 0:
            print(k)
        b.to_csv(outfile,
                   mode = "a",
                   header = k == 0)
    
    
    
    


##%%
#if __name__ == "__main__":
#    
#    while True:
#        #path = "/Users/vitorhadad/Documents/kidney/matching/data/"
#        try:
#            path = "data/"
#            name = str(np.random.randint(1e8))
#            data = prepare_mdp(1, time_length = 200)
#            pickle.dump(data, open(path + "data_with_value_{}.pkl".format(name), "wb"))
#        except:
#            pass
