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
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from os import listdir
from torch import autograd
import pickle
from random import choice, shuffle
from copy import deepcopy
from itertools import cycle
from torch.nn import SELU, ReLU
    
from matching.policy_function.policy_function_lstm import RNN
from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet

from matching.solver.kidney_solver2 import optimal, greedy

from matching.environment.abo_environment import ABOKidneyExchange
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
    
    if "GCN" in str(type(net)):
        X = env.X(t)[np.newaxis,:]
        A = env.A(t)[np.newaxis,:]
        logits, counts = net.forward(A, X)
        
    elif "MLP" in str(type(net)):  
        X = env.X(t)
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])[np.newaxis,:,:]
        logits, counts = net.forward(Z)
             
    elif "RNN" in str(type(net)): 
        X = env.X(t, graph_attributes = True, dtype = "numpy")
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])[np.newaxis,:,:].transpose((1,0,2))        
        logits, counts = net.forward(Z)            
    else:
        raise ValueError("Unknown algorithm")
        
    probs = F.softmax(logits.squeeze(), dim = 1)[:,1].data.numpy()
    #import pdb; pdb.set_trace()
    counts = np.round(counts.data.numpy()).astype(int).flatten()[0]
    #if counts % 2 == 1: counts += 1
    return probs, counts



def get_cycle_priors(living, cycles, probs):
    liv_idx = {j:i for i,j in enumerate(living)}
    cycle_probs = np.zeros(len(cycles))
    for k,cyc in enumerate(cycles):
        cycle_probs[k] = np.mean([probs[liv_idx[i]] for i in cyc])
    return cycle_probs


#%%


if __name__ == "__main__":
    

    from sys import platform
    from os import system
    from matching.solver.kidney_solver2 import get_cycles
    
    if platform == "linux":
        net_file = None
        time_length = 500
    else:
        net_file = None
        time_length = 1000
    
    
    burnin = 0
    entry_rate = 5
    death_rate = .1
    
    horizon = 10
    newseed = str(np.random.randint(1e8))
    train = True
    disc = 0.1
    
    net = torch.load("results/RNN_50-1-abo_4386504")

#%%   
    
    for k in [2]:
    
        
        print("Creating environment")
        env = ABOKidneyExchange(entry_rate, 
                                death_rate,
                                time_length,
                                seed = k)
        
        print("Solving environment")
        opt = optimal(env)
        gre = greedy(env)
        
        o = get_n_matched(opt["matched"], 0, env.time_length)
        g = get_n_matched(gre["matched"], 0, env.time_length)
        
        rewards = []
        actions = []
        t = -1
        print("Beginning")
        #%%
        for t in range(env.time_length):
            
            living = np.array(env.get_living(t))
            if len(living) == 1:
                continue
            
            probs, counts = evaluate_policy(net, env, t) 
            _, cycles = get_cycles(env, living)
            priors = get_cycle_priors(living, cycles, probs)
            idx = np.argsort(priors)
            selected = []
            for cyc in map(list, cycles):
                if cyc[0] not in selected and \
                    cyc[1] not in selected:
                    selected.extend(cyc) 
                if len(selected) >= counts:
                    break
            
            
            print("Size {}".format(len(living)), end = "\t")
            print("E[Prob] {:1.2f}".format(probs.mean()), end = "\t")
            print("Counts: {:d}".format(counts))        
            
#            s = optimal(env, 
#                      t_begin = t,
#                      t_end = t,
#                      subset = selected)
#               
#            
#            mat = s["matched"][t]
            
            r = len(selected)
            
            env.removed_container[t].update(selected)
            print(selected)
            #actions.append(s["matched_cycles"])
            rewards.append(r)
            
            
            msg = " Time:" + str(t) + \
                  " Reward" + str(r) + \
                  " Total:" + str(np.sum(rewards)) + \
                  " G:" +  str(g[:t].sum()) + \
                  " O:" +  str(o[:t].sum())
            
            print(msg)
            print(msg, file = open("RL_" + newseed + ".txt", "a"))
            
            
               
            if train and t % horizon == 0 and t >= (burnin + 2*horizon):
                
                net.optim.zero_grad()
                
                train_times = list(range(t-2*horizon, t-horizon))
                
                env_opt = deepcopy(env)
                
                for s in train_times:
                 
                    env_opt.removed_container[s].clear()
                
            
                for s in train_times:
                    
                    opt_m = optimal(env_opt, t_begin=s, t_end = s+2*horizon)
                    
                    baseline = disc_mean(get_n_matched(opt_m["matched"], s, s+2*horizon),
                                                       disc)
                    
                    rs = disc_mean(rewards[s:s+horizon], disc)
                    
                    adv = rs - baseline
                    
                    actions[s].reinforce(torch.FloatTensor(actions[s].size()).fill_(adv))
                    
                    env_opt.removed_container[s].update(env.removed_container[s])
                    
                    
                autograd.backward([actions[s] for s in train_times])
                
                net.optim.step()
                
            if t % 500 == 0 and t > 500:
                
                torch.save(net, "results/RL_" + newseed)
                
                
            if platform == "linux" and t > 500 and np.sum(rewards) < (g[:t].sum()*.8):
                system("qsub job_policy.pbs")
                exit()
        
        
        if platform != "linux":

            fig = plt.figure()
            plt.plot(cumavg(rewards), label ="this");
            plt.plot(cumavg(o), label = "opt");
            plt.plot(cumavg(g), label = "greedy");
            plt.legend()
            plt.title("{}, $\gamma$:{}"\
                      .format(str(net), disc))
            #fig.savefig("results/RL_" + newseed)
            
    