#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7th 17:44:35 2017

MCTS with policy function

"""


from copy import deepcopy
import numpy as np
import pandas as pd
from random import shuffle, choice
from itertools import chain
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from torch import nn, cuda
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

from matching.solver.kidney_solver2 import  optimal, greedy
from matching.utils.data_utils import get_additional_regressors, clock_seed
from matching.utils.env_utils import snapshot, remove_taken
from matching.utils.data_utils import flatten_matched, disc_mean , get_n_matched




def get_cycles(env, t, n_times):
    
    cycles = set()
    
    for _ in range(n_times):
        
        cs = optimal(env, t, t)["matched_cycles"][t]
        cycles.update([tuple(cyc) for cyc in cs])
        
    cycles = list(cycles)
    shuffle(cycles)
        
    return cycles


def evaluate_priors(net, env, t, cycles):
    n = len(cycles)
    if n == 1:
        return np.array([1])
    else:
        p = evaluate_policy(net, env, t)
        
        priors = np.zeros(n)
        for k, cyc in enumerate(cycles):
            if cyc is not None:
                i, j = cyc
                priors[k] = p.loc[i] * p.loc[j] + 1e-5

        priors /= np.sum(priors)
        
        return priors  


def evaluate_prior_order(net, env, t, cycles):
    priors = evaluate_priors(net, env, t, cycles)
    return np.flip(np.argsort(priors).flatten(),0).tolist()


def next_nonoverlapping_cycle(cur, cycles):
    try:
        return next(c for c in cycles if set(c).isdisjoint(cur))
    except StopIteration:
        return []
        





def get_actions(net, env, t, n_times = 50):
    
    cycles = get_cycles(env, t, n_times)
    order = evaluate_prior_order(net, env, t, cycles)
    actions = [[]]
    for i in range(1, len(order)+1):
        next_cycle = next_nonoverlapping_cycle(actions[-1], cycles)
        if len(next_cycle):
            actions.append(actions[-1] + list(next_cycle))
        else:
            break
        
    for a in actions:
        assert len(set(a)) == len(a)
        
    return tuple([tuple(a) for a in actions])


#%%



class Node:
    
    def __init__(self, 
                 parent,
                 t,
                 reward,
                 env,
                 taken,
                 actions):
        
        
        self.reward = reward
        self.env = env
        self.visits = 0
        self.children = []
        self.parent = parent
        self.t = t
        self.taken = taken
        self.actions = tuple(actions)
        self.expandable = set(actions)
        

    def next_action(self):
        try:
            return self.expandable.pop()
        except:
            import pdb; pdb.set_trace()
    
        
    def update(self, reward):
        self.reward += reward
        self.visits += 1
    
    
    def is_fully_expanded(self):
        return len(self.children) == len(self.actions)
            
            
    def __repr__(self):
        return "\nTaken: {} \nt: {}"\
                "\nChildren: {}"\
                "\nVisits: {} \nReward: {}"\
                  "\nActions: {}"\
            .format(self.taken,
                    self.t,
                    len(self.children),
                    self.visits, 
                    self.reward,
                    self.actions)


      
def run(root,
        scalar,
        tree_horizon,
        rollout_horizon,
        n_rollouts,
        net = None,
        gamma = 0.97):
    
    node = tree_policy(root,
                       root.t + tree_horizon,
                       net,
                       scalar)
    
    
    if node.taken is not None:
        r = []
        for i in range(n_rollouts):
            r.append(rollout(node.parent.env,
                             node.t,
                             node.t + rollout_horizon,
                             node.taken,
                             gamma))
        r = np.mean(r)
    
    backup(node, r)
    
    
    
    
    
def tree_policy(node, tree_horizon, net, scalar):
    while node.t < tree_horizon:
        if not node.is_fully_expanded():
            return expand(node, net)
        else:
            node = best_child(node, net, scalar)
    return node

        
        
    
def expand(node, net):        
    action = node.next_action()
    if action is None:
        child = advance(node, net)
    else:
        child = stay(node, action)
    node.children.append(child)
    return child  



def best_child(node, net, scalar):   
    if scalar is None:     
        return choice(node.children)

    else:
        rewards = np.array([c.reward for c in node.children])
        visits = np.array([c.visits for c in node.children])

        scores = compute_score(rewards, visits, priors = 1, scalar = scalar)
        argmaxs = np.argwhere(scores == np.max(scores)).flatten()
        chosen = np.random.choice(argmaxs)
                
        return node.children[chosen]


    
        
def backup(node, reward):
     while node != None:
        node.visits += 1
        node.reward += reward
        node = node.parent
        
        
        
def compute_score(rewards, visits, priors, scalar):
    N = sum(visits)
    exploit = rewards / visits
    explore = priors * np.sqrt(np.log(N)/visits)    
    scores = exploit + scalar*explore
    return scores
    
    

def advance(node, net):
    """Used when parent node chooses None or its last action"""
    child_env = snapshot(node.env, node.t)
    child_t = node.t + 1
    child_env.populate(child_t, child_t + 1)
    child_acts = get_actions(net, child_env, child_t)
    return Node(parent = node,
                t = child_t,
                env = child_env,
                reward = 0,
                taken = None,
                actions = child_acts)



def stay(node, taken):
    """Used when parent chooses an action that is NOT None"""
    child_t = node.t
    child_env = snapshot(node.env, node.t)
    child_env.removed_container[child_t].update(taken)
    child_acts = remove_taken(node.actions, taken)
    return Node(parent = node,
                t = child_t,
                env = child_env, 
                taken = taken,
                actions = tuple(child_acts),
                reward = 0)



def choose(root, criterion):
    
    shuffle(root.children)
    
    print("Choosing")
    for c in root.children:
        print("Option:", c.taken,
              " Visits: ", c.visits,
              " Avg reward: %1.3f" % (c.reward/c.visits),
              " Expl: %1.3f" % np.log(root.visits/c.visits))
    
    if criterion == "visits":
        most_visits = max([c.visits for c in root.children])
        most_visited_children = [c for c in root.children if c.visits == most_visits]
        # Break ties with avg rewards
        best = max(most_visited_children,
                   key = lambda c: c.reward/c.visits)
        
    elif criterion == "rewards":
        best = max(root.children,
               key = lambda c: c.reward/c.visits)
        
    return best.taken
    


def count_matched_today(env, t_begin, t_end, taken, n_times = 10):
    snap = snapshot(env, t_begin)
    matched_today = 0
    
    for i in range(n_times):
        if t_begin + 1 < t_end:
            snap.populate(t_begin + 1, t_end)
        opt_mt = optimal(snap, t_begin, t_end)["matched_cycles"][t_begin]
        matched_today += taken in opt_mt
        
    return matched_today / n_times
    



def rollout(env, t_begin, t_end, taken, gamma = 0.97):

    snap = snapshot(env, t_begin)
    snap.populate(t_begin+1, t_end, seed = clock_seed())
    snap.removed_container[t_begin].update(taken)
    
    value = greedy(snap, t_begin+1, t_end)
    matched = get_n_matched(value["matched"], t_begin, t_end)
    matched[0] = len(taken)
    
    return disc_mean(matched,  gamma)
    










def mcts(env, 
         t, 
         net = None,
         criterion = "visits",
         scl = 1,
         tpa = 5,
         tree_horizon = None,
         rollout_horizon = None,
         n_rolls = 1,
         gamma = 0.97):
    
    
    if tree_horizon is None:
        tree_horizon = int(1/env.death_rate)
    if rollout_horizon is None:
        rollout_horizon = int(1/env.death_rate)
    
    
    root = Node(parent = None,
                t = t,
                reward = 0,
                env = snapshot(env, t),
                taken = None,
                actions = get_actions(net, env, t))
    
    
    #print("Actions: ", root.actions)
    n_act = len(root.actions)
    if n_act > 1:    
        a = choice(root.actions)
        n_iters = int(tpa * n_act)
         
        for i_iter in range(n_iters):
            
            run(root,
                scalar = scl,
                tree_horizon = tree_horizon,
                rollout_horizon = rollout_horizon,
                net = net,
                n_rollouts = n_rolls,
                gamma = gamma)
            
            
        a = choose(root, criterion)

        print("Ran for", n_iters, "iterations and chose:", a)

    else:
        
        a = root.actions[0]
        print("Chose the only available action:", a)

    return a


