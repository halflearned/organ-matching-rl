#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 09:24:27 2017

@author: vitorhadad
"""



import numpy as np
from itertools import chain
import torch
from torch import autograd

from matching.policy_function.policy_function_gcn import GCNet
from matching.policy_function.policy_function_mlp import MLPNet
from matching.solver.kidney_solver2 import optimal, greedy
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.utils.data_utils import get_additional_regressors




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




def evaluate_policy(net, env, t):
    if "GCN" in str(type(net)):
        X = env.X(t)[np.newaxis,:]
        A = env.A(t)[np.newaxis,:]
        yhat = net.forward(A, X)
        
    elif "MLP" in str(type(net)):  
        X = env.X(t)
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])
        yhat = net.forward(Z)
        
    else:
        raise ValueError("Unknown algorithm")
        
    return yhat




def finish_episode(net, saved_actions, episode_rewards, gamma = 1):
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
    
    from os import listdir
    from random import choice
    
    net_file = choice([f for f in listdir("results/") if
                       f.startswith("GCN") or 
                       f.startswith("MLP")])
    burnin = 100
    entry_rate = 5
    death_rate = .1
    time_length = 200
    newseed = str(np.random.randint(1e8))


    if net_file is not None:
        net = torch.load("results/" + net_file)
        net.train()
    else:
        net= None
        
#%%
    for i in range(1000):

        env = SaidmanKidneyExchange(entry_rate,
                                    death_rate,
                                    time_length)
    
        g = greedy(env)
        o = optimal(env)
        
        ms = []
        episode_rewards = []
        pg_rewards = []
        actions = []
        probs = []
        
        for t in range(time_length):
            
            living = np.array(env.get_living(t))
            if len(living) > 0:
                a_probs = evaluate_policy(net, env, t) 
            else:
                continue

            
            a = a_probs.squeeze(0).bernoulli()
            selected = a.data.numpy().flatten().astype(bool)
            
            matched = living[selected]
            env.removed_container[t].update(matched)
            
            
            if t > burnin:
                
                actions.append(a)
                
                n_dead = len([n for n,d in env.nodes.data() 
                              if d["death"] == t 
                              and n not in env.removed(t)])
    
                r = np.full_like(selected, 
                                 fill_value = -n_dead,
                                 dtype = np.float32)
                
                pg_rewards.append(r)
                
                episode_rewards.append(np.sum(selected))
                
                
                
        finish_episode(net, actions, pg_rewards)
    
        print("Matched (this): \t", np.sum(episode_rewards))
        print("Matched (greedy): \t", g["obj"])
        print("Matched (opt): \t", o["obj"])
        
        torch.save(net, "RL_" + newseed + "_" + net_file)
        
    
    
    
    
