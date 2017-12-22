#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:24:27 2017

@author: vitorhadad
"""



import numpy as np
from itertools import chain
import torch
import torch.nn.functional as F
import matplotlib
from tqdm import trange
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from os import listdir
import pandas as pd
from torch import autograd
import pickle
from random import choice, shuffle
from copy import deepcopy
from itertools import cycle
from torch.nn import SELU, ReLU
from sys import argv

from matching.utils.env_utils import two_cycles    

from matching.policy_function.policy_function_rnn import RNN

from matching.policy_function.policy_function_agcn import AGCNet
from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet
from matching.solver.kidney_solver2 import optimal, greedy
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange

from matching.utils.data_utils import get_additional_regressors
from matching.tree_search.mcts import mcts
from matching.utils.data_utils import get_n_matched, clock_seed
from matching.utils.data_utils import disc_mean, cumavg

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#%%

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
#%%


if __name__ == "__main__":
    

    from sys import platform
    from os import system
    
    if platform == "linux":
        net_file = None
        time_length = 1000
    else:
        net_file = None
        time_length = 100
    
    
    burnin = 0
    entry_rate = 5
    death_rate = .1
    
    horizon = 50
    newseed = str(np.random.randint(1e8))
    train = True
    disc = 0.99
    
    net = torch.load("results/policy_function_lstm")
    
    
    def best_cycle(env, t, thres = 0):
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
#%%   
    print("Creating environment")
    env = OPTNKidneyExchange(entry_rate, 
                         death_rate,
                         time_length,
                         seed = 12345)
    
    opt = optimal(env)
    gre = greedy(env)
        
    o = get_n_matched(opt["matched"], 0, env.time_length)
    g = get_n_matched(gre["matched"], 0, env.time_length)
    
    for i in range(1000):
        
        env.removed_container.clear()
        
        print("Solving environment")
        
        rewards = []
        actions = []
        t = 0
        
        np.random.seed(clock_seed())
        thresholds = np.random.uniform(0, 1, size = 5)
       
        #thresholds= np.zeros(5)
        print("Beginning")
        #%%
        for t in trange(env.time_length):
    
            r = 0
            while True:
                
                n = len(env.get_living(t))
                
                if n < 10:
                    thres = thresholds[0]
                elif n < 20:
                    thres = thresholds[1]
                elif n < 30:
                    thres = thresholds[2]
                elif n < 40:
                    thres = thresholds[3]
                elif n < 50:
                    thres = thresholds[4]
                    
                a = best_cycle(env, t, thres = thres)
                
                if a is None:
                    break
                else:
                    r += len(a)
                    env.removed_container[t].update(a)
            
            rewards.append(r)
                    
            
            msg = " Thresholds:" + " ".join([str(s.round(2)) for s in thresholds]) + \
                  " Time:" + str(t) + \
                  " Size:" + str(len(env.get_living(t))) + \
                  " Reward:" + str(r) + \
                  " Total:" + str(np.sum(rewards)) + \
                  " G:" +  str(g[:t].sum()) + \
                  " O:" +  str(o[:t].sum())

        
        
        if platform != "linux":
    
            fig = plt.figure()
            plt.plot(cumavg(rewards));
            plt.plot(cumavg(o));
            plt.plot(cumavg(g));
            #plt.title("lstm with threshold: {}".format(threshold))
            plt.ylim((2.5, 4))
            fig.savefig("results/RL_" + newseed)
            
            fig, ax = plt.subplots(1)
            pd.Series(rewards).rolling(200).mean().plot(ax = ax)
            pd.Series(o[:t]).rolling(200).mean().plot(ax = ax)
            pd.Series(g[:t]).rolling(200).mean().plot(ax = ax)
            

        print(msg)
        print(msg, file = open("results/rnn_results.txt", "a"))
            