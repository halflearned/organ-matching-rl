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
import multiprocessing as mp

from matching.solver.kidney_solver2 import  optimal
from matching.utils.data_utils import get_additional_regressors, clock_seed
from matching.utils.env_utils import get_actions, snapshot, remove_taken
from matching.utils.data_utils import flatten_matched



class Node:
    
    def __init__(self, 
                 parent,
                 t,
                 reward,
                 env,
                 taken,
                 actions,
                 priors = None):
        
        
        self.reward = reward
        self.env = env
        self.visits = 0
        self.children = []
        self.parent = parent
        self.t = t
        self.taken = taken
        self.actions = tuple(actions)
        self.expandable = set(actions)
        self.priors = priors
        

    def next_action(self):
        return self.expandable.pop()
    
        
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
        net = None):
    
    #import pdb; pdb.set_trace()
    
    node = tree_policy(root,
                       root.t + tree_horizon,
                       net,
                       scalar)
    
    
    if node.taken is not None:
        r = parallel_rollout(node,
                             rollout_horizon,
                             n_rollouts)    
    else:
        r = 1
    
    backup(node, r)
    
    
    
def tree_policy(node, tree_horizon, net, scalar):
    while node.t < tree_horizon:
        if not node.is_fully_expanded():
            return expand(node)
        else:
            node = best_child(node, net, scalar)
    return node

        
        
    
def expand(node):        
    action = node.next_action()
    if action is None:
        child = advance(node)
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
        
        if net is not None:
            priors = evaluate_priors(net, node.env, node.t, node.actions)
        else:
            priors = 1
            
        scores = compute_score(rewards, visits, priors, scalar)
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
    
    

def advance(node):
    """Used when parent node chooses None or its last action"""
    child_env = snapshot(node.env, node.t)
    child_t = node.t + 1
    child_env.populate(child_t, child_t + 1)
    child_acts = get_actions(child_env, child_t)
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
    


    
def parallel_rollout(node, horizon, n):   
    try:
        prcs = mp.cpu_count() 
        with mp.Pool(processes = prcs) as pool:             
            results = [pool.apply_async(rollout,
                            args = (node.parent.env,
                                    node.t,
                                    node.t + horizon, 
                                    node.taken))
                        for i in range(n)]
            res = [r.get() for r in results]
    
    except Exception:
        print("Error during parallel rollout. Fallback on regular loop.")
        res = []
        for i in range(n):
            #env = deepcopy(node.parent.env)
            res.append(rollout(snapshot(node.parent.env, node.t),
                               node.t,
                               node.t + horizon,
                               node.taken))
    return np.mean(res)
    
    
def rollout(env, t_begin, t_end, taken):
    seed = clock_seed()
    rem = deepcopy(env.removed_container)
    loss_leave = simulate_unmatched_dead(env, t_begin, t_end, seed)
    loss_take = simulate_unmatched_dead(env, t_begin, t_end, seed, taken)
    env.removed_container = rem
    return (1 + loss_leave)/(1 + loss_take)
    

    
def simulate_unmatched_dead(env, t_begin, t_end, seed=None, taken=None):
    if taken is not None:
        env.removed_container[t_begin].update(taken)
    env.populate(t_begin + 1, t_end, seed=seed)
    solution = optimal(env, t_begin=t_begin, t_end=t_end)
    matched = flatten_matched(solution["matched"])
    if taken is not None:
        matched.update(taken)
    dead = get_dead(env, matched, t_begin, t_end)
    return len(dead)



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


    
def evaluate_policy(net, env, t):
    try:
        if "GCN" in str(type(net)):
            X = env.X(t)[np.newaxis,:]
            A = env.A(t)[np.newaxis,:]
            yhat = net.forward(A, X)
            
        elif "MLP" in str(type(net)):  
            X = env.X(t)
            G, N = get_additional_regressors(env, t)
            Z = np.hstack([X, G, N])
            yhat = net.forward(Z)
            
    except Exception as e:
        import pdb; pdb.set_trace()
        
    return pd.Series(index = env.get_living(t),
                     data = yhat\
                                .data\
                                .numpy()\
                                .flatten())


def evaluate_priors(net, env, t, actions):
    n = len(actions)
    if n == 1:
        return np.array([1])
    else:
        p = evaluate_policy(net, env, t)
        none_idx = actions.index(None)
        
        priors = np.zeros(n)
        for k, cyc in enumerate(actions):
            if cyc is not None:
                i, j = cyc
                priors[k] = p.loc[i] * p.loc[j] + 1e-5
    
        priors[none_idx] = 1/n     
        priors /= (priors.sum() * n/(n-1))    
   
        return priors  

