#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:13:45 2017

@author: vitorhadad
"""

import pickle
import pandas as pd
import numpy as np
import torch

from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.solver.kidney_solver import KidneySolver
from matching.utils.data_utils import get_additional_regressors
from matching.utils.env_utils import get_actions



    
def get_node_probs(net, env, t):
    index = env.get_living(t)
    A = env.A(t)    
    X = env.X(t)
    G, N = get_additional_regressors(env, t)
    Z = np.hstack([X, G, N])
    node_probs = net.forward(A, Z)
    return index, node_probs
    



gcn = (50, 1)
policy_path = "results/policy_{}_{}.pkl".format(*gcn)
net = pickle.load(file = open(policy_path, "rb"))["net"]



t = 0
env = SaidmanKidneyExchange(5, .1, 100)
idx, nps = get_node_probs(net, env, t)
unord_actions = np.array(get_actions(env, t))













