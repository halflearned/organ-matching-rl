#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:33:52 2017

@author: vitorhadad
"""

from copy import deepcopy
import numpy as np
import pandas as pd
from random import shuffle, choice
from collections import Counter
import pickle
from sys import argv

import torch
from torch import nn, cuda
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from matching.utils.data_utils import get_additional_regressors
from matching.utils.data_utils import get_n_matched, clock_seed
from matching.utils.data_utils import disc_mean, cumavg


from matching.environment.optn_environment import OPTNKidneyExchange
from matching.solver.kidney_solver2 import optimal_with_discount, optimal, greedy
from matching.utils.data_utils import  clock_seed, cumavg
from matching.utils.env_utils import snapshot, remove_taken, get_loss
from matching.utils.data_utils import flatten_matched, disc_mean , get_n_matched
#%%



def best_cycle(net, env, t, thres = 0):
    liv_idx = dict({j:i for i,j in enumerate(env.get_living(t))})
    if len(liv_idx) == 0:
        return None
    a_probs = evaluate_policy(net, env, t) 
    cycles = list(map(tuple, optimal(env, t, t)["matched_cycles"][t])) 
    if len(cycles) == 0:
        return None
    np_probs = np.zeros(len(cycles))
    for q,(i,j) in enumerate(cycles):
        np_probs[q] = np.mean((a_probs[liv_idx[i]], a_probs[liv_idx[j]]))
    selected = np.argmax(np_probs)
    if np_probs[selected] >= thres:
        return cycles[selected]
    else:
        return None
    
    
def evaluate_policy(net, env, t):
    
    if "AGCN" in str(type(net)):
        X = env.X(t)
        G,N = get_additional_regressors(env, t)
        A = env.A(t)
        yhat = net.forward(A, X, G, N)
        
    elif "GCN" in str(type(net)):
        X = env.X(t)[np.newaxis,:]
        A = env.A(t)[np.newaxis,:]
        yhat = net.forward(A, X)
        
    elif "MLP" in str(type(net)):  
        X = env.X(t)
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])
        yhat = net.forward(Z)
             
    elif "RNN" in str(type(net)): 
        n = len(env.get_living(t))
        lens = [n]
        #lens = lens.tolist()#torch.LongTensor(lens)
        
        X = env.X(t)
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])
        er = np.full((Z.shape[0], 1), fill_value = env.entry_rate)
        dr = np.full((Z.shape[0], 1), fill_value = env.death_rate)
        ZZ = np.hstack([Z, er, dr])[np.newaxis,:,:].transpose((1,0,2))
        ZZv = Variable(torch.FloatTensor(ZZ))
        
        ZZv = pack_padded_sequence(ZZv, lens)
        outputs,_ = net.forward(ZZv)  
        p = net.out_layer(outputs[:n,0])
        yhat = F.softmax(p, dim= 1)[:,1]   
        
    else:
        raise ValueError("Unknown algorithm")
        
    return yhat


def greedy_actions(env, t, n_repeats):
    return set([tuple(optimal(env, t, t)["matched"][t]) 
                for _ in range(n_repeats)])
    
def simulate_value(snap, t, horizon, gamma, n_iters):
    values = []
    for i in range(n_iters):
        snap.populate(t+1, t+horizon+1, seed = clock_seed())  
        opt = optimal_with_discount(snap, t, t+horizon, gamma = gamma)
        values.append(opt["obj"])
    return values
    

def predict(env, t, horizon, gamma, n_iters, n_repeats = 20):
    g_acts = greedy_actions(env, t, n_repeats)
    if len(g_acts) == 0:
        return 0
    values = dict()
    for g in g_acts:
        snap = snapshot(env, t)
        snap.removed_container[t].update(g_acts)
        avg_value = np.mean(simulate_value(snap, t, horizon, 
                                       gamma, n_iters))
        values[g] = avg_value
    return values


#%%

env = OPTNKidneyExchange(5, .1, 100, seed = 12345)

opt = optimal(env)
gre = greedy(env)
o = get_n_matched(opt["matched"], 0, env.time_length)
g = get_n_matched(gre["matched"], 0, env.time_length)

#%%
time_limit = env.time_lenth
t = 0
horizon = 10 #int(argv[1]) 
n_iters = 1 #int(argv[2])
gamma = 1 #float(argv[3])
n_times = 5
rewards = []
matched = set()       
rndname = str(np.random.randint(1e8))
outfilename = "{}_{}_{}_{}.txt".format(horizon, 
               n_times, gamma, rndname)

#%%

while t < time_limit:

    c = predict(env, t, horizon,  gamma = gamma, n_iters = n_iters)
    a = max(c)
    env.removed_container[t].update(a)
    rewards.append(len(a))
    
    print(np.mean(rewards), 
          np.mean(g[:t+1]),
          np.mean(o[:t+1]),
          file = open(outfilename, "a"))
    
    t += 1
    
    with open(outfilename, "wb") as f:
        pickle.dump(obj = {"rewards":rewards,
                     "o":o[1:t+1],
                     "g":g[1:t+1]},
                    file = f)
 
#%%
#plt.plot(cumavg(rewards), linewidth = 2);
#plt.plot(cumavg(g[1:t+1]));
#plt.plot(cumavg(o[1:t+1]));
#plt.ylim(3.5, 5)
    
    

    
    
    

