#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:44:35 2017
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
                 scalar = 0.7071,
                 n_parallel_rollouts = 1):
        
        self.mcl = max_cycle_length    
        self.tree_horizon = int(t + tree_horizon)
        self.rollout_horizon = int(rollout_horizon)
        self.n_prl = n_parallel_rollouts
        
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
        
        score = self.compute_score(rewards, visits)
        argmax = np.argwhere(score == np.max(score)).flatten()
        chosen = np.random.choice(argmax)
                
        return node.children[chosen]
    
    
        
    def backup(self, node, reward):
    	 while node != None:
            node.visits += 1
            node.reward += reward
            node = node.parent
    	
        
        
    def compute_score(self, rewards, visits):
        N = sum(visits)
        exploit = rewards / visits
        explore = np.sqrt(np.log(N)/visits)	
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
        g_obj = solver.greedy(env, t_begin = t_init, t_end = T)["obj"]
      
        r = 0
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
                
        reward = float(r > g_obj*.85)
        return reward
    
    
    
    
    def choose(self):
        shuffle(self.root.children)
        print("Choosing")
        for c in self.root.children:
            print("Taken:", c.taken, " Visits: ", c.visits, "Reward: ",  c.reward)
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
    from sys import argv
    
  
    print("Using:")
    for i,arg in enumerate(argv):
        print("arg[",i,"]:",arg)
        

    er =  5
    dr =  .1
    
    if len(argv) > 1:
        scl = float(argv[1])
        time_per_action = int(argv[2])
        prl = int(argv[3])
        t_horiz = int(argv[4])
        r_horiz = int(argv[5])
        time_length = 100
    else:    
        scl = 1.4142
        prl = 1
        t_horiz = 4
        r_horiz = 6
        time_per_action = 1
        time_length = 10
    
    #%%
    opt = None
    g   = None
   
    while True:
        
        seed = clock_seed()
        
        name = str(seed)
        
        def write(*s):
            print(*s)
    
    
        env = SaidmanKidneyExchange(entry_rate  = er,
                                    death_rate  = dr,
                                    time_length = time_length,
                                    seed = seed)
        
       
        matched = defaultdict(list)
        rewards = 0                
            
        t = 0
        while t < env.time_length:
            
            write("\nStarting ", t)
            mc = MCTS(env, t,
                      tree_horizon = t_horiz,
                      rollout_horizon = r_horiz,
                      scalar = scl,
                      n_parallel_rollouts = prl)
        
            iters = 0
            # Spawn all jobs
            write("Actions: ", mc.root.actions)
            n_act = len(mc.root.actions)

            if n_act > 1:    
                t_end = time() + time_per_action * n_act**2
                while time() < t_end:                  
                    mc.run()
                    iters += 1
                a = mc.choose()
            else:
                a = mc.root.actions[0]
                
            write("Ran for", iters, "iterations and chose", a)
        
            if a is not None:
                write("Staying at t.")
                assert a[0] not in env.removed_container[t]
                assert a[1] not in env.removed_container[t]
                env.removed_container[t].update(a)
                matched[t].extend(a)
                rewards += len(a)
            else:
                write("Done with", t, ". Moving on to next period\n")
                t += 1
                
           
            results = [seed,er,dr,t_horiz,r_horiz,
                       scl,time_per_action,prl,
                       time_length,rewards,g,opt]
            write("Current results:", results)
        
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
        
        results = [seed,er,dr,t_horiz,r_horiz,
                       scl,time_per_action,prl,
                       time_length,rewards,g,opt]
        

        with open("mcts_plain_results.txt", "a") as f:
            s = ",".join([str(s) for s in results])
            f.write(s + "\n")
