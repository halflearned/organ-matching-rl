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
from time import time    
import multiprocessing as mp

from itertools import chain

from matching.solver.kidney_solver2 import  optimal, greedy
from matching.utils.data_utils import get_additional_regressors, clock_seed
from matching.utils.env_utils import get_actions, snapshot, remove_taken
from matching.tree_search.mcts_with_opt_rollout import Node, run




def best_child(node, net, scalar):        
    return choice(node.children)


def choose(root):
    shuffle(root.children)
    print("Choosing")
    for c in root.children:
        print("Option:", c.taken,
              " Visits: ", c.visits,
              " Avg reward: %1.3f" % (c.reward/c.visits))
    
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
    
    