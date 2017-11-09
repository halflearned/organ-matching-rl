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



#%%
    


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
                 actions):
        
        
        self.reward = reward
        self.env = env
        self.visits = 1
        self.children = []
        self.parent = parent
        self.t = t
        self.taken = taken
        self.actions = tuple(actions)
        self.expandable = set(actions)
        self.priors = evaluate_priors(env, t, actions)
        

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

        
#%%
class MCTS:
    
    def __init__(self, env, t,
                 max_cycle_length = 2,
                 tree_horizon = 4,
                 rollout_horizon = 6,
                 scalar = 1.41,
                 n_parallel_rollouts = 1,
                 use_priors = True,
                 adversary = "opt",
                 deflator = 0.9):
        
        self.mcl = max_cycle_length    
        self.tree_horizon = int(t + tree_horizon)
        self.rollout_horizon = int(rollout_horizon)
        self.n_prl = n_parallel_rollouts
        self.use_priors = use_priors
        self.adversary = adversary
        self.deflator = deflator
        
        child_env = deepcopy(env)
        child_env.erase_from(t+1)
        acts = get_actions(env, t)
        self.root = Node(parent = None,
                         env = child_env,
                         t = t,
                         reward = 0,
                         taken = None,
                         actions = acts)
        self.scalar = scalar
       
        
        
    def run(self):
        node = self.tree_policy(self.root)
        r = self.parallel_rollout(node)
        self.backup(node, r)
        
        
        
    def tree_policy(self, node):
        while node.t < self.tree_horizon:
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node
    
        
        
    
    def expand(self, node):        
        action = node.next_action()
        if action is None:
            child = self.advance(node)
        else:
            child = self.stay(node, action)
        node.children.append(child)
        return child  
    
    
    
    def best_child(self, node):        
            
        rewards = np.array([c.reward for c in node.children])
        visits = np.array([c.visits for c in node.children])
        
        if self.use_priors:
            scores = self.compute_score(rewards, visits, node.priors)
        else:
            scores = self.compute_score(rewards, visits, 1)
        argmaxs = np.argwhere(scores == np.max(scores)).flatten()
        chosen = np.random.choice(argmaxs)
                
        return node.children[chosen]
    
    
        
    def backup(self, node, reward):
         while node != None:
            node.visits += 1
            node.reward += reward
            node = node.parent
        
        
        
    def compute_score(self, rewards, visits, priors = 1):
        N = sum(visits)
        exploit = rewards / visits
        explore = priors * np.sqrt(np.log(N)/visits)    
        scores = exploit + self.scalar*explore
        return scores
        
            
        
    def remove_taken(self, actions, taken):
        return [e for e in actions
                    if e is None or 
                    len(set(e).intersection(taken)) == 0]
    
    
    

    def advance(self, node):
        """Used when parent node chooses None or its last action"""
        child_env = deepcopy(node.env)
        child_t = node.t + 1
        child_env.populate(child_t+1, child_t + 2)
        child_acts = get_actions(child_env, child_t)
        return Node(parent = node,
                    t = child_t,
                    env = child_env,
                    reward = 0,
                    taken = None,
                    actions = child_acts)



    def stay(self, node, taken):
        """Used when parent chooses an action that is NOT None"""
        child_t = node.t
        child_env = deepcopy(node.env)
        child_env.removed_container[child_t].update(taken)
        child_acts = self.remove_taken(node.actions, taken)
        return Node(parent = node,
                    t = child_t,
                    env = child_env, 
                    reward = len(taken),
                    taken = taken,
                    actions = tuple(child_acts))
        
        

    def rollout(self, node):
        env = node.env 
        t_init = node.t
        T = t_init + self.rollout_horizon
        env.populate(t_init + 1, T, seed = clock_seed())
        solver = KidneySolver(2)
        
        if self.adversary == "opt":
            adv_obj = solver.greedy(env, t_begin = t_init, t_end = T)["obj"]
        elif self.adversary == "greedy":
            adv_obj = solver.solve(env, t_begin = t_init, t_end = T)["obj"]
        else:
            raise NotImplementedError("Unknown adversary!")
        
        
        r = 0#node.latent_reward
        for t in range(t_init, T):
            actions = get_actions(env, t)
            while len(actions) > 0:
                a = actions.pop()
                if a is not None:
                    actions = self.remove_taken(actions, a)
                    env.removed_container[t].update(a)
                    r += len(a)
                else:
                    break
                
        reward = float(r > adv_obj*self.deflator)
        return reward
    
    
    
    
    def choose(self):
        shuffle(self.root.children)
        print("Choosing")
        for c in self.root.children:
            print("Option:", c.taken, " Visits: ", c.visits, "Reward: ",  c.reward)
        best = max(self.root.children, key = lambda x: x.visits)
        return best.taken
    
    
          
    def parallel_rollout(self, node):
        prcs = min(mp.cpu_count(), self.n_prl)
        
        with mp.Pool(processes = prcs) as pool:     
            
            results = [pool.apply_async(
                        self.rollout, args = (node,))
                        for i in range(self.n_prl)]
            res = [r.get() for r in results]
            
            
            
        return np.mean(res)
    
    
    
    
# Slightly faster
def two_cycles(env, t):
    nodes = list(env.get_living(t))
    cycles = []
    for i, u in enumerate(nodes):
        for w in nodes[i:]:
            if env.has_edge(u,w) and env.has_edge(w,u):
                cycles.append((u,w))
    return cycles

#%%
if __name__ == "__main__":
    
    from collections import defaultdict
    from itertools import chain
    import pickle
    from sys import argv

    er =  5
    dr =  .1
    time_length = 150
    
    scl = float(argv[1])
    tpa = float(argv[2])
    prl = int(argv[3])
    t_horiz = int(argv[4])
    r_horiz = int(argv[5])
    gcn = eval(argv[6])
    defl = float(argv[7])
    use_priors = eval(argv[8])

    print("scl", scl)
    print("tpa", tpa)
    print("prl", prl)
    print("t_horiz", t_horiz)
    print("r_horiz", r_horiz)
    print("gcn", gcn)
    print("defl",defl)
    print("use_priors", use_priors)

    config = (scl, tpa, prl, t_horiz, r_horiz, gcn, defl, use_priors)

    policy_path = "results/policy_{}_{}.pkl".format(*gcn)
    net = pickle.load(file = open(policy_path, "rb"))["net"]
    
    
    def evaluate_policy(env, t):
        A = env.A(t)    
        X = env.X(t)
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])
        return pd.Series(index = env.get_living(t),
                         data = net.forward(A, Z).data.numpy())



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
                    i,j = cyc
                    priors[k] = p.loc[i] * p.loc[j] + 1e-5
        
            priors /= priors.sum() * (n-1)/n
            priors[none_idx] = 1/n 
            if not np.all(np.isfinite(priors)):
                import pdb; pdb.set_trace()
            
            return priors
        
            
    for seed in range(10000, 15000):
     
        opt = None
        g   = None
    
        seed = seed #clock_seed()

        name = str(seed)        

        env = SaidmanKidneyExchange(entry_rate  = er,
                death_rate  = dr,
                time_length = time_length,
                seed = seed)

        matched = defaultdict(list)
        rewards = 0                
    
        t = 0
        while t < env.time_length:
            
            print("\nStarting ", t)
            mc = MCTS(env, t,
              tree_horizon = t_horiz,
              rollout_horizon = r_horiz,
              scalar = scl,
              n_parallel_rollouts = prl,
              use_priors = use_priors,
              deflator = defl)
    
            iters = 0
        
            print("Actions: ", mc.root.actions)
            n_act = len(mc.root.actions)
    
            if n_act > 1:    
                t_end = time() + tpa * n_act**2
                while time() < t_end:                  
                    mc.run()
                    iters += 1
                    a = mc.choose()
            else:
                a = mc.root.actions[0]
            
            print("Ran for", iters, "iterations and chose", a)
    
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
            

        all_matched = list(chain(*matched.values()))
        assert len(all_matched) == len(set(all_matched))


    
        env = SaidmanKidneyExchange(entry_rate  = er,
                death_rate  = dr,
                time_length = time_length,
                seed = seed)
    
        solver = KidneySolver(2)
        opt = solver.optimal(env)["obj"]
        g = solver.greedy(env)["obj"]
    
        print("MCTS rewards: ", rewards)
        print("GREEDY rewards:", g)
        print("OPT rewards:", opt)
    
        results = [seed,er,dr,time_length,*config,rewards,g,opt]
    

        with open("results/mcts_with_policy_results.txt", "a") as f:
            s = ",".join([str(s) for s in results])
            f.write(s + "\n")
    
