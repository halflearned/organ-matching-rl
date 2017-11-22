#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:44:35 2017

Adversarial MCTS: Reward is zero up to some depth D,
where the agent then "wins" if they do strictly 
better than greedy. Rewards are diff btw MC and greedy.

adv_results: # of parallel searches is fixed at 10
adv_results2: # of parallel searches varies
    
"""

from copy import deepcopy
import numpy as np
from random import choice, shuffle
from time import time	
import multiprocessing as mp

from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.solver.kidney_solver import KidneySolver

def clock_seed():
    return int(str(int(time()*1e8))[10:])    


# Slightly faster
def two_cycles(env, t):
    nodes = list(env.get_living(t))
    cycles = []
    for i, u in enumerate(nodes):
        for w in nodes[i:]:
            if env.has_edge(u,w) and env.has_edge(w,u):
                cycles.append((u,w))
    return cycles



class Node():
	
    def __init__(self, 
                 parent,
                 t,
                 reward,
                 depth,
                 taken = None,
                 matched = None,
                 actions = None,
                 expandable = None,
                 max_cycle_length = 2):
        
        self.env = env
        self.visits = 1
        self.reward = reward
        self.children = []
        self.parent = parent
        self.t = t
        self.depth = depth
        self.taken = taken
        self.mcl = max_cycle_length
        self.matched = matched or set()
        self.actions = actions or self.get_actions()
        self.expandable = expandable or self.actions.copy()


    def pop_action(self):
        return self.expandable.pop()

        
    def update(self,reward):
        self.reward += reward
        self.visits += 1
        
        
    def get_actions(self):
        cycles = two_cycles(self.env, self.t) 
        actions = list(map(tuple, cycles))
        actions.append(None)
        shuffle(actions)
        return actions
    
    
    def is_fully_expanded(self):
        return len(self.expandable) == 0
            
            
    def __repr__(self):
        return "\nDepth: {} \nChildren: {}"\
                "\nVisits: {} \nReward: {}"\
                  "\nTaken: {} \nt: {}"\
                  "\nActions: {} \nExpandable: {}"\
                  "\nMatched :{}"\
            .format(self.depth,
                    len(self.children),
                    self.visits, 
                    self.reward,
                    self.taken,
                    self.t,
                    self.actions,
                    self.expandable,
                    self.matched)

		

class MCTS:
    
    def __init__(self, env, t,
                 max_cycle_length = 2,
                 max_depth = 4,
                 horizon = 10,
                 iterations = 100,
                 scalar = 1/np.sqrt(2),
                 use_parallel_rollout = True,
                 n_parallel_rollouts = 10):
        
        self.env = deepcopy(env)
        self.iterations = iterations
        self.mcl = max_cycle_length    
        self.max_depth = max_depth
        self.horizon = horizon
        self.root = Node(parent = None,         
                         t = t,
                         reward = 0,
                         taken = None,
                         depth = 0)
        self.scalar = scalar
        self.use_prl = True
        self.n_prl = n_parallel_rollouts 


    def backup(self, node, reward): 
    	while node != None:
            node.visits += 1
            node.reward += reward
            node = node.parent
    	
        
            
    def best_child(self, node):
        
        if len(node.children) == 0:
            #import pdb; pdb.set_trace()
            raise ValueError("No children!")
            
        rewards = np.array([c.reward for c in node.children])
        visits = np.array([c.visits for c in node.children])
        
        exploit = rewards / visits
        explore = np.sqrt(2*np.log(node.visits)/visits)	
        score = exploit + self.scalar*explore
            
        argmax = np.argwhere(score == np.max(score)).flatten()
        chosen = np.random.choice(argmax)
                
        return node.children[chosen]



    def stay(self, node, action):
        """Used when parent chooses an action that is NOT None"""        
        return Node(parent = node,
                    t = node.t,
                    depth = node.depth,
                    reward = len(action),
                    taken = action,
                    matched = node.matched.copy() | set(action),
                    max_cycle_length = node.mcl,
                    expandable = [e for e in node.expandable
                                  if e is None or 
                                  len(set(e).intersection(action)) == 0])
        
        
        
    def advance(self, node):
        """Used when parent node chooses None"""
        return Node(parent = node,
                    t = node.t + 1,
                    depth = node.depth + 1,
                    reward = 0,
                    taken = None,
                    matched = node.matched.copy(),
                    max_cycle_length = node.mcl)



    def expand(self, node):
        
        action = node.pop_action()
        if action is None:
            child = self.advance(node)
        else:
            child = self.stay(node, action)
        node.children.append(child)
        return child


    def get_actions(self, env, t):
        cycles = two_cycles(env, t) 
        actions = list(map(tuple, cycles))
        actions.append(None)
        shuffle(actions)
        return actions
        
        
        
    def rollout(self, node):
        
        env = deepcopy(node.env)
        t_init = node.t
        env.populate(t_init, t_init + self.horizon, seed = clock_seed())
        solver = KidneySolver(2)
        g_obj = solver.greedy(env)["obj"]
        
        r = 0
        for t in range(t_init, t_init + self.horizon):
            actions = self.get_actions(env, t)
            a = self.rollout_policy(actions)
            if a is not None:
                env.removed_container[t].update(a)
                r += len(a)
            
        reward = r - g_obj

        return reward
            
    
    def rollout_policy(self, actions):
        return choice(actions)
    
    
    def tree_policy(self, node):
        while node.depth != self.max_depth:
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node
    
      
    def parallel_rollout(self, node):
        with mp.Pool(processes=mp.cpu_count()) as pool:         
            results = [pool.apply_async(
                        self.rollout, args = (node,),
                        )
                        for i in range(self.n_prl)]
            res = [r.get() for r in results]
        
        return np.mean(res)
    
    
    def search(self):
        for i in range(self.iterations):
            node = self.tree_policy(self.root)
            if self.use_prl:
                r = self.parallel_rollout(node)
            else:
                r = self.rollout(node)
            self.backup(node, r)
        shuffle(self.root.children)
        best = max(self.root.children,  key = lambda x: x.visits)
        return best.taken
        
    
        
    

    
# Slightly faster
def two_cycles(env, t):
    nodes = list(env.get_living(t))
    cycles = []
    for i, u in enumerate(nodes):
        for w in nodes[i:]:
            if env.has_edge(u,w) and env.has_edge(w,u):
                cycles.append((u,w))
    return cycles



    
    
if __name__ == "__main__":
    
    
    from collections import defaultdict
    from itertools import chain
    from sys import argv
        
    print("Using:")
    for i,arg in enumerate(argv):
        print("arg[",i,"]:",arg)
        
    if len(argv) > 1:
        er =  int(argv[1])
        dr =  float(argv[2])
        mxd = int(argv[3])
        hor = int(argv[4])
        scl = float(argv[5])
        its = int(argv[6])
        prl = int(argv[7])
    else:
        er =  5
        dr =  .1
        mxd = 6
        hor = 4
        scl = .2
        its = 20
        prl = 1
        
        
    time_length = 100
    
    
    for _ in range(1):
        
        seed = clock_seed()
        
        name = "66892704" #str(seed)
    
        env = SaidmanKidneyExchange(entry_rate  = er,
                                    death_rate  = dr,
                                    time_length = time_length,
                                    seed = seed)
       
        matched = defaultdict(list)
        rewards = 0                
            
        t = 0
        while t < env.time_length:
            print(t)
            mc = MCTS(env, t,
                      max_depth = mxd, 
                      horizon = hor,
                      scalar = scl,
                      iterations = its)
            
            a = mc.search()
            if a is not None:
                assert a[0] not in env.removed_container[t]
                assert a[1] not in env.removed_container[t]
                env.removed_container[t].update(a)
                matched[t].extend(a)
                rewards += len(a)
            else:
                t += 1
                if t % 5 == 0: print(t)
        
        all_matched = list(chain(*matched.values()))
        assert len(all_matched) == len(set(all_matched))
        
        
            
    #%%
        
        env = SaidmanKidneyExchange(entry_rate  = er,
                                    death_rate  = dr,
                                    time_length = time_length,
                                    seed = seed)
        solver  = KidneySolver(2)
        opt = solver.optimal(env)
        g = solver.greedy(env)
        
        print("MCTS rewards: ", rewards)
        print("GREEDY rewards:", g["obj"])
        print("OPT rewards:", opt["obj"])
        
        results = [seed,er,dr,mxd,hor,scl,its,prl,time_length,
                   rewards,g["obj"],opt["obj"]]
        

#        with open("adv_results.txt", "a") as f:
#            s = ",".join([str(s) for s in results])
#            f.write(s + "\n")
#        
        
        
        
        
        
        
        
        
        