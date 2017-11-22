#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7th 17:44:35 2017

MCTS with policy function

"""

from copy import deepcopy
import numpy as np
import pandas as pd
from random import shuffle
from time import time    
import multiprocessing as mp

from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.solver.kidney_solver import KidneySolver
from matching.utils.data_utils import get_additional_regressors




def clock_seed():
    return int(str(int(time()*1e8))[10:])    


def get_actions(env, t):
    cycles = two_cycles(env, t) 
    actions = list(map(tuple, cycles))
    actions.append(None)
    shuffle(actions)
    return actions


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
        self.visits = 1
        self.children = []
        self.parent = parent
        self.t = t
        self.taken = taken
        self.actions = tuple(actions)
        self.expandable = set(actions)
        self.priors = priors
        

    def next_action(self):
        return self.expandable.pop()
    
        
    def update(self,reward):
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
        use_priors,
        n_rollouts):
    
    #import pdb; pdb.set_trace()
    
    node = tree_policy(root,
                       root.t + tree_horizon,
                       use_priors,
                       scalar)
    
    if node.taken is not None:
        r = parallel_rollout(node,
                             rollout_horizon,
                             n_rollouts)    
    else:
        r = 1
    
    backup(node, r)
    
    
    
def tree_policy(node, tree_horizon, use_priors, scalar):
    while node.t < tree_horizon:
        if not node.is_fully_expanded():
            return expand(node)
        else:
            node = best_child(node, use_priors, scalar)
    return node

        
        
    
def expand(node):        
    action = node.next_action()
    if action is None:
        child = advance(node)
    else:
        child = stay(node, action)
    node.children.append(child)
    return child  



def best_child(node, use_priors, scalar):        
        
    rewards = np.array([c.reward for c in node.children])
    visits = np.array([c.visits for c in node.children])
    
    if use_priors:
        priors = evaluate_priors(node.env, node.t, node.actions)
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
    child_env = deepcopy(node.env)
    child_t = node.t + 1
    child_env.populate(child_t, child_t + 1)
    child_acts = get_actions(child_env, child_t)
    return Node(parent = node,
                t = child_t,
                env = child_env,
                reward = 1,
                taken = None,
                actions = child_acts)



def stay(node, taken):
    """Used when parent chooses an action that is NOT None"""
    child_t = node.t
    child_env = deepcopy(node.env)
    child_env.removed_container[child_t].update(taken)
    child_acts = remove_taken(node.actions, taken)
    return Node(parent = node,
                t = child_t,
                env = child_env, 
                taken = taken,
                actions = tuple(child_acts),
                reward = 1)



def choose(root):
    shuffle(root.children)
    print("Choosing")
    for c in root.children:
        print("Option:", c.taken,
              " Visits: ", c.visits,
              " Avg reward: ", c.reward/c.visits)
    
    most_visits = max([c.visits for c in root.children])
    most_visited_children = [c for c in root.children if c.visits == most_visits]
    # Break ties with avg rewards
    best = max(most_visited_children,
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
            env = deepcopy(node.parent.env)
            res.append(rollout(env,
                               node.t,
                               node.t + horizon,
                               node.taken))
    return np.mean(res)
    
    
#%%
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
    solver = KidneySolver(2)
    solution = solver.solve(env, t_begin=t_begin, t_end=t_end)
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


#%%

def remove_taken(actions, taken):
    return [e for e in actions
            if e is None or 
                len(set(e).intersection(taken)) == 0]



def two_cycles(env, t):
    nodes = list(env.get_living(t))
    cycles = []
    for i, u in enumerate(nodes):
        for w in nodes[i:]:
            if env.has_edge(u,w) and env.has_edge(w,u):
                cycles.append((u,w))
    return cycles


def evaluate_priors(env, t, actions):
    n = len(actions)
    if n == 1:
        return np.array([1])
    else:
        p = evaluate_policy(env, t)
        none_idx = actions.index(None)
        
        priors = np.zeros(n)
        for k, cyc in enumerate(actions):
            if cyc is not None:
                i, j = cyc
                priors[k] = p.loc[i] * p.loc[j] + 1e-5
    
        priors[none_idx] = 1/n     
        priors /= (priors.sum() * n/(n-1))    
   
        return priors  


def flatten_matched(m, burnin = 0):
   return set(chain(*[x for t,x in m.items() if t >= burnin]))


#%%
if __name__ == "__main__":
    
    from collections import defaultdict
    from itertools import chain
    from random import choice
    import pickle
    import torch
    from sys import platform
    from scipy.stats import geom

    er = 5
    dr = .1
    
    
    for episode in range(1000):
        if platform == "darwin":
            scl = .5
            tpa = 2
            t_horiz = 2
            r_horiz = 30
            n_rolls = 1
            gcn = 'test.pkl'
            use_priors = False
            time_length = 25
            burnin = 0
        else:
            scl = np.random.uniform(0.1, 3)
            tpa = choice([2, 5, 10])
            t_horiz = choice([2, 5, 10])
            r_horiz = np.random.randint(1, geom(.1).ppf(.95))
            n_rolls = np.random.randint(1, 5)
            gcn = choice([(5,1), (50,1)])
            use_priors = choice([True, False])
            time_length = 200
            burnin = 100
        
        if burnin > time_length:
            raise ValueError("burnin > T!")
        
        
        print("USING:")
        print("scl", scl)
        print("tpa", tpa)
        print("t_horiz", t_horiz)
        print("r_horiz", r_horiz)
        print("n_rolls", n_rolls)
    
        config = (scl, tpa, n_rolls, t_horiz, r_horiz, gcn, use_priors)
    
        if use_priors:
            net = torch.load(gcn)
        
        
        def evaluate_policy(env, t):
            #A = env.A(t)    
            X = env.X(t)
            G, N = get_additional_regressors(env, t)
            Z = np.hstack([X, G, N])
            return pd.Series(index = env.get_living(t),
                             data = net.forward(Z)\
                                        .data\
                                        .numpy()\
                                        .flatten())
          
        
     
        opt = None
        g   = None
    
        seed = clock_seed()

        name = str(seed)        

        env = SaidmanKidneyExchange(entry_rate  = er,
                death_rate  = dr,
                time_length = time_length,
                seed = seed)


        matched = defaultdict(list)
        rewards = 0                
#%%    
        t = 0
        while t < env.time_length:
            
            print("Now at", t,
                  file = open(name + ".txt", "w"))
        
            print("\nStarting ", t)
            root = Node(parent = None,
                        t = t,
                        reward = 0,
                        env = env,
                        taken = None,
                        actions = get_actions(env, t))

            iters = 0
        
            print("Actions: ", root.actions)
            n_act = len(root.actions)
    
            if n_act > 1:    
                
                n_iters = int(tpa * n_act)
                
                for i_iter in range(n_iters):
                    
                    run(root,
                        scalar = scl,
                        tree_horizon = t_horiz,
                        rollout_horizon = r_horiz,
                        use_priors = use_priors,
                        n_rollouts = n_rolls)
                    
                a = choose(root)
                print("Ran for", n_iters, "iterations and chose:", a)
        
            else:
                
                a = root.actions[0]
                print("Chose the only available action:", a)
    
            
            if a is not None:
                
                print("Staying at t.")
                assert a[0] not in env.removed_container[t]
                assert a[1] not in env.removed_container[t]
                env.removed_container[t].update(a)
                matched[t].extend(a)
                rewards += len(a)
            
            else:
            
                print("Done with", t, ". Moving on to next period\n")
                t += 1
        
        
#%%
                
        this_matched = flatten_matched(matched)

        env = SaidmanKidneyExchange(entry_rate  = er,
                death_rate  = dr,
                time_length = time_length,
                seed = seed)
    
        solver = KidneySolver(2)
        opt = solver.optimal(env)#["obj"]
        greedy = solver.greedy(env)#["obj"]
    
        g_matched = flatten_matched(greedy["matched"], burnin)
        opt_matched = flatten_matched(opt["matched"], burnin)
        
        n = len(env.get_living(burnin, env.time_length))
        
        g_loss = len(get_dead(env, g_matched, burnin))/n
        opt_loss = len(get_dead(env, opt_matched, burnin))/n
        this_loss = len(get_dead(env, this_matched, burnin))/n
    
        print("MCTS loss: ", this_loss)
        print("GREEDY loss:", g_loss)
        print("OPT loss:", opt_loss)
        
        
        results = [seed,er,dr,time_length,*config,this_loss,g_loss,opt_loss]
    

        with open("results/mcts_with_opt_rollout_results3.txt", "a") as f:
            s = ",".join([str(s) for s in results])
            f.write(s + "\n")
    
        