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

from matching.solver.kidney_solver2 import  optimal, greedy
from matching.utils.data_utils import get_additional_regressors, clock_seed
from matching.utils.env_utils import snapshot, remove_taken
from matching.utils.data_utils import flatten_matched, disc_mean , get_n_matched


def get_actions(env, t, n_times = 100):
    
    sols = []
    
    for _ in range(n_times):
        
        opt_m = optimal(env, t, t)["matched_pairs"]
        if len(opt_m) == 0:
            break
        sols.append(tuple(sorted(opt_m)))
        
    return list(set(sols)) + [()]
    
    


def evaluate_policy(net, env, t):
    if net is None:
        yhat = np.ones_like(env.get_living(t))
    
    elif "GCN" in str(type(net)):
        X = env.X(t)[np.newaxis,:]
        A = env.A(t)[np.newaxis,:]
        yhat = net.forward(A, X).data.numpy().flatten()
        
    elif "MLP" in str(type(net)):  
        X = env.X(t)
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])
        yhat = net.forward(Z).data.numpy().flatten()
    
    return pd.Series(index = env.get_living(t),
                     data = yhat)




def evaluate_priors(net, env, t, actions):
    
    action_priors = np.zeros(len(actions))
    
    node_priors = evaluate_policy(net, env, t)
    
    for k, a in enumerate(actions):
 
        if len(a) == 0:
            action_priors[k] = max(np.mean(node_priors) - 0.1*len(actions), 0)
        
        else:
            action_priors[k] = np.mean(node_priors.loc[list(a)])
        
    return action_priors





class Node:
    
    def __init__(self, 
                 parent,
                 t,
                 reward,
                 env,
                 taken,
                 actions,
                 priors = 1):
        
        
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
        net = None,
        gamma = 0.97):
    
    node = tree_policy(root,
                       root.t + tree_horizon,
                       net,
                       scalar)
    

    rollout_reward = np.mean([rollout(node.parent.env,
                         node.t,
                         node.t + rollout_horizon,
                         node.taken,
                         gamma) for _ in range(n_rollouts)])
    
    backup(node, rollout_reward)
    

    
    
    
    
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

    if len(node.children) == 1:
        return node.children[0]

    else:
        rewards = np.array([c.reward for c in node.children])
        visits = np.array([c.visits for c in node.children])
        node_actions = [c.taken for c in node.children]
        priors = evaluate_priors(net, node.env, node.t, node_actions)
        
        scores = compute_score(rewards, visits, priors, scalar)
        argmaxs = np.argwhere(np.isclose(scores, np.max(scores), 1e-3)).flatten()
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



def choose(root, criterion = "visits"):
    
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
    



def rollout(env, t_begin, t_end, taken, gamma):

    snap = snapshot(env, t_begin)
    snap.populate(t_begin+1, t_end, seed = clock_seed())
    snap.removed_container[t_begin].update(taken)
    
    opt = optimal(snap, t_begin+1, t_end)
    opt_matched = get_n_matched(opt["matched"], t_begin, t_end)
    opt_matched[0] = len(taken)
    opt_value = disc_mean(opt_matched,  gamma)
    
#    g = greedy(snap, t_begin+1, t_end)
#    g_matched = get_n_matched(g["matched"], t_begin, t_end)
#    g_matched[0] = len(taken)
#    g_value = disc_mean(g_matched,  gamma)
    
    r = opt_value #- g_value
    
    return r








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
                actions = get_actions(env, t))
    
    
    
    print("Actions: ", root.actions)
    n_act = len(root.actions)
    if n_act > 1:    

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
        #import pdb; pdb.set_trace()
        
        print("Ran for", n_iters, "iterations and chose:", a ,"\n\n\n")

    else:
        
        a = root.actions[0]
        print("Chose the only available action:", a,"\n\n\n")

    return a


