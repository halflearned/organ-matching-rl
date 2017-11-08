#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:24:27 2017

@author: vitorhadad
"""


import numpy as np
from collections import defaultdict
from itertools import chain
from tqdm import trange

from matching.solver.kidney_solver import KidneySolver
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.utils.data_utils import get_additional_regressors

def mdp(net, thres = .5, seed = None):
    
  
    env = SaidmanKidneyExchange(entry_rate  = 5, #choice([2, 5, 10]),
                                death_rate  = .1, #choice([0.1, 0.01, 0.005]),
                                time_length = 100,
                                seed = seed)

    solver = KidneySolver(2)  
    
    opt = solver.optimal(env)
    greedy = solver.greedy(env)
    
    rs = []
    ms = []
    for t in trange(env.time_length):
        
        idx = np.array(env.get_living(t))
        A = env.A(t)
        X = env.X(t)
        G, N = get_additional_regressors(env, t)
        XX = np.hstack([X, G, N])
        probs = net.forward(A, XX).data.numpy() 
        chosen = idx[probs >= thres]
        sol = solver.solve_subset(env, chosen)
        m = list(chain(*sol["matched"].values()))
        env.removed_container[t].update(m)
        rs.append(len(m))
        ms.extend(m)

    output = {"opt":opt["obj"],
              "greedy":greedy["obj"],
              "matched": ms,
              "obj": sum(rs),
              "rs": rs}
    
    return output




#%%
if __name__ == "__main__":
    
    import pickle
    import torch.nn.functional as F
    from torch import nn, optim
    from sys import argv
    from time import time
    
    from matching.utils.make_data import merge_data
    from matching.gcn.gcn import GraphConvNet
    
    
    #%% Get data
    data = merge_data("X", "A", "G", "N", "y", path = "data/")
    X_shape = data["X"][0].shape[1]
    N_shape = data["N"][0].shape[1]
    G_shape = data["G"][0].shape[1]
    
    #%%
    
    if len(argv) == 1:
        gcn_size = 10
        num_layers = 3
    else:
        gcn_size = int(argv[1])
        num_layers = int(argv[2])
        
    
    name = str(np.random.randint(1e8))
        
    net = GraphConvNet(feature_size = X_shape + G_shape + N_shape,
                       gcn_size = gcn_size,
                       num_layers = num_layers,
                       dropout_prob = .2,
                       output_fn = F.sigmoid,
                       loss_fn = nn.BCELoss,
                       opt = optim.Adam,
                       opt_params = dict(lr = 0.001))
        

    training_losses = []
    

    
    #%%
    n_hours = 11.5
    t_begin = time()
    t_end = t_begin + n_hours * 3600
    while time() < t_end:
    
        for k, (A, *XX, y) in enumerate(zip(data["A"], data["X"], data["G"], data["N"], data["y"])):
            
            XX = np.hstack(XX)
            loss, out = net.run(A.toarray(), XX, y)
            training_losses.append(loss)
            if k % 100 == 0: print(np.array(training_losses)[-20:].mean())
    
        net.eval()
        pickle.dump(obj = {"net": net,
                           "training_losses": training_losses,
                           "gcn_size": gcn_size,
                           "num_layers": num_layers},
                    file = open("results/policy_{}_{}.pkl".format(gcn_size, num_layers), "wb"))
            
        
    elapsed = time() - t_begin
    print("Finished. Elapsed: {:1.0f} seconds ({:1.2f} hours)".format(elapsed, elapsed / 3600))