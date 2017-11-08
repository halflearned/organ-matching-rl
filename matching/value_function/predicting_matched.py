#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:24:27 2017

@author: vitorhadad
"""


from kidney_solver import KidneySolver
import numpy as np
import pandas as pd
import networkx as nx
from saidman_environment import SaidmanKidneyExchange
from itertools import chain


def relabel_env(env):
    n = env.number_of_nodes()
    new_labels = dict(zip(env.nodes(), np.random.permutation(n)))
    ret_labels = {l:i for i,l in new_labels.items()}
    rel_env = env.relabel_nodes(new_labels)
    return rel_env, ret_labels
    

def relabel_sol(sol, labels):
    rsol = sol.copy()
    rsol["matched"] = {}
    for t,ms in sol["matched"].items():
        rsol["matched"][t] = [labels[x] for x in ms]
    return rsol
        

def learn(X, y, algo, sampling = None):
    
    if sampling == "under":
        nnz = np.count_nonzero(y)
        i_nonzeros = np.argwhere(y).flatten()
        zeros = np.argwhere(1-y).flatten()
        i_zeros = np.random.choice(zeros, size = 2*nnz)
        idx = np.hstack((i_nonzeros, i_zeros))
        X = X[idx]
        y = y[idx]
    elif sampling == "over":
        nz = len(y) - np.count_nonzero(y)
        i_nonzeros = np.argwhere(y).flatten()
        zeros = np.argwhere(1-y).flatten()
        i_zeros = np.random.choice(zeros, size = nz)
        idx = np.hstack((i_nonzeros, i_zeros))
        X = X[idx]
        y = y[idx]
        
    
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
    
    algo.fit(X_train, y_train)
    y_pred = algo.predict(X_test)
    return algo, y_pred, y_test


def reg_matrix(env, t, fns):
    
    X = env.X(t)
    
    if fns:
        Xp = []
        for fn in fns:
            subg = nx.subgraph(env, env.get_living(t))
            stat = fn(subg)
            stat = np.array(list(stat.values())).reshape(-1, 1)
            Xp.append(stat)
        Xp = np.hstack(Xp)
        X = np.hstack([X, Xp])
    return X



def generate_data(n_sets, fncs = None):
    
    Xs = []
    ys = []
    
    for i in tqdm(range(n_sets)):
        
        env = SaidmanKidneyExchange(entry_rate  = 5, #choice([2, 5, 10]),
                                    death_rate  = .1, #choice([0.1, 0.01, 0.005]),
                                    time_length = 150,
                                    seed = i)
    
        solver = KidneySolver(max_cycle_length = 2, #choice([2,3]),
                              burn_in = 75)  
        
        opt = solver.optimal(env)
        t = int(np.median(list(opt["matched"].keys())))
        
        Xs.append(reg_matrix(env, t, fncs))
    
        ms = opt["matched"][t]
        
        living = env.get_living(t)
        y = np.isin(np.array(living), ms)
        y = y.astype(np.float32).reshape(-1, 1)
         
        ys.append(y)

    
    ys = np.vstack(ys).ravel()
    Xs = np.vstack(Xs)
    
    return Xs, ys


def step(env, solver, t, nodelist, requires_reindex = True):

    if requires_reindex:
        nodelist, _ = env.reindex_to_absolute(nodelist, t)
    
    removable = set()
    
    # Removal by matching
    m = solver.solve_subset(env, nodelist)
    matched = list(chain(*m["matched"].values()))
    removable.update(matched)
    
    # Removal by death
    dead = env.get_dying(t)
    removable.update(dead)
    
    li = env.removed(t)
    try:
        for i in removable: assert i not in li 
    except AssertionError:
        import pdb; pdb.set_trace()
    
    # Drop
    env.removed_container[t].update(removable)
    
    return len(matched)



def mdp(algo, n_episodes = 10):
    
    output = []

    for i in tqdm(range(1000, 1000 + n_episodes),
                  desc = "Running MDP"):
        
        env = SaidmanKidneyExchange(entry_rate  = 5, #choice([2, 5, 10]),
                                    death_rate  = .1, #choice([0.1, 0.01, 0.005]),
                                    time_length = 150,
                                    seed = i)
    
        solver = KidneySolver(max_cycle_length = 2, #choice([2,3]),
                              burn_in = 75)  
        
        opt = solver.optimal(env)
        greedy = solver.greedy(env)
        
        t = int(np.median(list(opt["matched"].keys())))
        
        for t in range(solver.burn_in):
            env.removed_container[t].update(opt["matched"][t])
        
        rs = []
        for t in range(solver.burn_in, env.time_length):
            
            XX = reg_matrix(env, t, functions)
            y_pred = algo.predict(XX)
            chosen = np.argwhere(y_pred).flatten()
            r = step(env, solver, t, chosen, True)
            rs.append(r)
    
        output.append([str(algo),
                        opt["obj"], 
                      greedy["obj"],
                      sum(rs)])
    
    return pd.DataFrame(data = output, 
                        columns = ["algo",
                                   "OPT",
                                   "GREEDY",
                                   "THIS"])



#%%
if __name__ == "__main__":
    
    #from random import choice
    #import pickle
    from tqdm import tqdm
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from random import choice
    from itertools import product
    from sklearn.neural_network import MLPClassifier
    
    functions = [nx.betweenness_centrality,
                 nx.out_degree_centrality,
                 nx.in_degree_centrality]
    
    X, y = generate_data(200, functions)
    
    #%%
    # Testing on static test data
    samplings = ["over", "under", None]
    thresholds = np.linspace(0.05, .95, 19)
    
    algos = [SVC(C = .1),
            SVC(C = 1),
            SVC(C = 10),
            SVC(C = 1000),
            SVC(C = 10000),
            GaussianProcessClassifier(),
            RandomForestClassifier(n_estimators = 100),
            RandomForestClassifier(n_estimators = 500),
            RandomForestClassifier(n_estimators = 1000),
            RandomForestClassifier(n_estimators = 5000),
            GradientBoostingClassifier(n_estimators = 100),
            GradientBoostingClassifier(n_estimators = 500),
            GradientBoostingClassifier(n_estimators = 1000),
            GradientBoostingClassifier(n_estimators = 5000),
            LogisticRegression(penalty = "l1", C = 100),
            LogisticRegression(penalty = "l1", C = 1),
            LogisticRegression(penalty = "l1", C = .01),
            LogisticRegression(penalty = "l2", C = 100),
            LogisticRegression(penalty = "l2", C = 1),
            LogisticRegression(penalty = "l2", C = .01),
            MLPClassifier(hidden_layers_sizes = [5]*1) ,
            MLPClassifier(hidden_layers_sizes = [5]*2) ,
            MLPClassifier(hidden_layers_sizes = [5]*3) ,
            MLPClassifier(hidden_layers_sizes = [5]*4) ,
            MLPClassifier(hidden_layers_sizes = [5]*5) ,
            MLPClassifier(hidden_layers_sizes = [5]*10) ,
            MLPClassifier(hidden_layers_sizes = [10]*1) ,
            MLPClassifier(hidden_layers_sizes = [10]*2) ,
            MLPClassifier(hidden_layers_sizes = [10]*3) ,
            MLPClassifier(hidden_layers_sizes = [10]*4) ,
            MLPClassifier(hidden_layers_sizes = [10]*5) ,
            MLPClassifier(hidden_layers_sizes = [10]*10) ,
            MLPClassifier(hidden_layers_sizes = [20]*1) ,
            MLPClassifier(hidden_layers_sizes = [20]*2) ,
            MLPClassifier(hidden_layers_sizes = [20]*3) ,
            MLPClassifier(hidden_layers_sizes = [20]*4) ,
            MLPClassifier(hidden_layers_sizes = [20]*5) ,
            MLPClassifier(hidden_layers_sizes = [20]*10) ,
            MLPClassifier(hidden_layers_sizes = [40]*1) ,
            MLPClassifier(hidden_layers_sizes = [40]*2) ,
            MLPClassifier(hidden_layers_sizes = [40]*3) ,
            MLPClassifier(hidden_layers_sizes = [40]*4) ,
            MLPClassifier(hidden_layers_sizes = [40]*5) ,
            MLPClassifier(hidden_layers_sizes = [40]*10)]

    outputs = []
    for algo, samp in product(algos, samplings):
        
        algo, y_pred, y_test = learn(X, y, algo, sampling = samp)
        tn, fp, fn, tp = confusion_matrix(y_test.ravel(), 
                                          y_pred.ravel()).ravel()
          
        print(algo)
        print("sum(y_pred)", np.sum(y_pred))
        print("sum(y_test)", np.sum(y_test))
        print("\nConfusion matrix")
        print("TP/(TP+FP) = ", tp/(tp + fp))
        print("TN/(TN+FN) = ", tn/(tn + fn))
        print("FP/(TP+FP) = ", fp/(tp + fp))
        print("FN/(TN+FN) = ", fn/(tn + fn))
        
        for thres in thresholds:
            out = mdp(algo, thres, n_episodes = 20)
            out["sampling"] = samp
            out["threshold"] = thres
            outputs.append(out)
    
    table = pd.concat(outputs, axis = 0, ignore_index = True)
    table.to_csv("predicting_matched.csv")
    
    
