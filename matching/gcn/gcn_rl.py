#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:24:27 2017

@author: vitorhadad
"""



import numpy as np
import pandas as pd
import networkx as nx
from itertools import chain
import torch
from gcn import to_var
from torch import autograd
import matplotlib
import pickle
from copy import deepcopy

from matching.solver.kidney_solver import KidneySolver
from matching.environment.saidman_environment import SaidmanKidneyExchange
matplotlib.use("Agg")


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
        

def sample(A, X, y, sampling = None):
    
    y = y.flatten()
    
    if sampling == "under":
        nnz = np.count_nonzero(y)
        i_nonzeros = np.argwhere(y).flatten()
        i_zeros = np.argwhere(1-y).flatten()
        i_zeros_resampled = np.random.choice(i_zeros, size = nnz)
        idx = np.hstack((i_nonzeros, i_zeros_resampled))
        A = A[idx, idx]
        X = X[idx]
        y = y[idx]
        
    elif sampling == "over":
        nz = len(y) - np.count_nonzero(y)
        i_nonzeros = np.argwhere(y).flatten()
        i_zeros = np.argwhere(1-y).flatten()
        i_nonzeros_resamp = np.random.choice(i_nonzeros, size = nz)
        idx = np.hstack((i_nonzeros_resamp, i_zeros))
        A = A[idx, idx]
        X = X[idx]
        y = y[idx]
    
    #A_train, A_test, X_train, X_test, y_train, y_test = train_test_split(
    #        A, X, y, test_size=0.3, random_state=42)
    y = y.reshape(-1, 1)

    return A, X, y




def generate_data(n_sets, fncs = None):
    
    Xs = []
    ys = []
    As = []
    
    for i in tqdm(range(n_sets), desc = "Generating Data"):
        
        env = SaidmanKidneyExchange(entry_rate  = 5, #choice([2, 5, 10]),
                                    death_rate  = .1, #choice([0.1, 0.01, 0.005]),
                                    time_length = 150,
                                    seed = i)
    
        solver = KidneySolver(max_cycle_length = 2, #choice([2,3]),
                              burn_in = 75)  
        
        opt = solver.optimal(env)
        t = int(np.median(list(opt["matched"].keys())))
        
        #Xs.append(reg_matrix(env, t, fncs))
        Xs.append(env.X(t))
        As.append(env.A(t))
    
        ms = opt["matched"][t]
        
        living = env.get_living(t)
        y = np.isin(np.array(living), ms)
        y = y.astype(np.float32).reshape(-1, 1)
         
        ys.append(y)

    
    return As, Xs, ys


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
    
    return matched


def prepare_mdp(seeds, 
                entry_rate = 5,
                death_rate = .1,
                time_length = 150,
                burn_in = 75,
                max_cycle_length = 2):
    
    data = {}
    
    for seed in tqdm(seeds, desc = "Preparing data"):
    
        env = SaidmanKidneyExchange(entry_rate  = entry_rate,
                                    death_rate  = death_rate,
                                    time_length = time_length,
                                    seed = seed)
    
        solver = KidneySolver(max_cycle_length = max_cycle_length,
                              burn_in = burn_in)  
        
        opt = solver.optimal(env)
        
        
        for t_match, m in opt["matched"].items():
            if t_match <= burn_in:
                env.removed_container[-1].update(m)
        
        greedy = solver.greedy(env, t_begin = burn_in)
    
        data[seed] = {"env": env,
                      "opt_obj": opt["obj"],
                      "greedy_obj": greedy["obj"],
                      "max_cycle_length": max_cycle_length,
                      "entry_rate": entry_rate,
                      "death_rate": death_rate,
                      "time_length": time_length,
                      "burn_in": burn_in}
        
        del solver # Needed?
    
    return data




def mdp(net, data):
    
    env = deepcopy(data["env"])
    
    solver = KidneySolver(max_cycle_length = data["max_cycle_length"],
                          burn_in = data["burn_in"]) 
    
    ms = []
    episode_rewards = []
    pg_rewards = []
    actions = []
    probs = []
    
    for t in range(solver.burn_in, env.time_length):
        
        A = env.A(t)
        X = env.X(t, "pandas")
        y = net.forward(A, X).unsqueeze(1)
        yhat = y.bernoulli()

        labels = X.index
        y_numpy = yhat.data.numpy().flatten()
        
        chosen = np.argwhere(y_numpy).flatten()
        probs.append(np.mean(y_numpy))
        
        m = step(env, solver, t, chosen, True)
        m_period = np.argwhere(np.isin(labels, m)).flatten()
        ms.extend(m)
        
        r = np.zeros_like(y_numpy, dtype = np.float32)
        r[m_period] = 1
        pg_rewards.append(r)
        
        episode_rewards.append(len(m))
        actions.append(yhat)
        
    finish_episode(net, actions, pg_rewards)

    print("Prob(Y = 1):", np.mean(probs))

    output = {"opt": data["opt_obj"],
              "greedy": data["greedy_obj"],
              "matched": ms,
              "obj": sum(episode_rewards)}
    
    return output



def finish_episode(net, saved_actions, episode_rewards, gamma = 1.1):
    R = torch.Tensor([0])
    feed_rewards = []
    for r in episode_rewards[::-1]:
        R = r + gamma * R.mean()
        
        feed_rewards.insert(0, R)
    
    for action, r in zip(saved_actions, feed_rewards):
        rr = torch.Tensor(r.reshape(-1,1))
        rr -= rr.mean()
        action.reinforce(rr)
    
    net.opt.zero_grad()
    autograd.backward(saved_actions, [None for _ in saved_actions])
    net.opt.step()


#%%
if __name__ == "__main__":
    
    #from random import choice
    #import pickle
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from gcn import GraphConvNet
    import torch.nn.functional as F
    from torch import nn, optim
    from random import choice
    import pickle

    lr = choice([0.01])
    seed = np.random.randint(1e8)
    name = str(seed)
    print(name)

    
    opt_ratio = []
    greedy_ratio = []
    test_opt_ratio = []
    test_greedy_ratio = []
    
    #%%
    
    
    training_data = pickle.load(open("training_data.pkl", "rb")) #prepare_mdp(range(n_training))
    test_data = pickle.load(open("test_data.pkl", "rb"))
    n_training = len(training_data)
    n_epochs = len(test_data)

    #%%
    
        
    net = GraphConvNet(feature_size = 10,
                       gcn_size = choice([1, 3, 5, 10, 50, 100]),
                       num_layers = choice([1, 5, 10]),
                       dropout_prob = choice([0, .1, .2, .5]),
                       output_fn = F.sigmoid,
                       loss_fn = nn.BCELoss,
                       opt = optim.Adam,
                       opt_params = dict(lr = lr),
                       seed = seed)
    
    print(net)
    
    #%%

    for epoch, (k, tests) in enumerate(test_data.items()):
        
        net.train()
        
        for i in range(n_training):
            
            print("\nEPOCH: ", epoch, "\tITERATION", i)    
            
            mdp_output = mdp(net, training_data[i])
            
            opt_ratio.append(mdp_output["obj"]/mdp_output["opt"])
            greedy_ratio.append(mdp_output["obj"]/mdp_output["greedy"])
            
            
            print("OPT: ", mdp_output["opt"])
            print("GREEDY: ", mdp_output["greedy"])
            print("THIS: ", mdp_output["obj"])
            print("OPT/THIS: %1.3f" % opt_ratio[-1])
            print("GREEDY/THIS: %1.3f" % greedy_ratio[-1])
            
       
        # Test
        net.eval()
        mdp_output = mdp(net, tests) 
        test_opt_ratio.append(mdp_output["obj"]/mdp_output["opt"])
        test_greedy_ratio.append(mdp_output["obj"]/mdp_output["greedy"])
        
        
        print("\nOPT: ", mdp_output["opt"])
        print("GREEDY: ", mdp_output["greedy"])
        print("THIS: ", mdp_output["obj"])
        print("OPT/THIS: %1.3f" % opt_ratio[-1])
        print("GREEDY/THIS: %1.3f" % greedy_ratio[-1])
        
        
        # PRINTING
        
        cfg_str = "L: {}  S: {}   D:{}    LR:{}".\
            format(net.num_layers, net.gcn_size, net.dropout_prob, lr)
        
        fig, ax = plt.subplots(1, 2, figsize = (15, 5))
        
        ax[0].plot(opt_ratio, label = "THIS/OPT")
        ax[0].plot(greedy_ratio, label = "GREEDY/OPT")
        ax[0].legend(loc = "best")
        ax[0].set_title("TRAIN " + cfg_str, fontsize = 14)
        
        ax[1].plot(test_opt_ratio, label = "THIS/OPT")
        ax[1].plot(test_greedy_ratio, label = "GREEDY/OPT")
        ax[1].legend(loc = "best")
        ax[1].set_title("TEST " + cfg_str, fontsize = 14)
        
        fig.savefig("ratios_" + name)
        np.save("test_" + name, np.array(test_opt_ratio))
        np.save("train_" + name, np.array(opt_ratio))
 
    
        print("\n\nAVG THIS/OPT: ", np.mean(opt_ratio[-n_training:]))
        print("\n\nAVG GREEDY/OPT: ", np.mean(greedy_ratio[-n_training:]))
        
#       
#
##%%
#import numpy as np
#from os import listdir
#import pandas as pd
#npys = [np.load(f) for f in listdir() if f.endswith(".npy")]
#n = max([len(r) for r in npys]) 
#
#rs = np.vstack([r for r in npys if len(r) == n])
#pd.DataFrame(rs.T).rolling(100).mean().plot()
#






#

