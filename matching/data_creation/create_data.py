#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:50:56 2018

@author: vitorhadad
"""

from sys import platform
from random import choice

import numpy as np
import pickle

from matching.utils.data_utils import clock_seed, get_features
from matching.environment.abo_environment import ABOKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange
    
while True:
    
    if platform=="linux":
        s = clock_seed()
        env_type = choice(["optn"])
        if env_type == "abo":
            env = ABOKidneyExchange(entry_rate=np.random.randint(5, 10),
                                    death_rate=np.random.choice([.1, .09, .08, .07, .06, .05]), time_length=1000,
                                    seed=s)
        elif env_type == "saidman":
            env = SaidmanKidneyExchange(entry_rate=np.random.randint(5, 10),
                                        death_rate=np.random.choice([.1, .09, .08, .07, .06, .05]), time_length=1000,
                                        seed=s)
        elif env_type == "optn":
            env = OPTNKidneyExchange(entry_rate = np.random.randint(5, 10),
                                death_rate = np.random.choice([.1,.09,.08,.07,.06,.05]), 
                                time_length = 1000, 
                                seed = s)
    else:
        env_type = "optn"
        s = 1234
        env = OPTNKidneyExchange(entry_rate=5,
                                death_rate=.1, 
                                time_length=1000, 
                                seed=1234)    
        
    X, G, E, Y = get_features(env)
    
    x_filename = "X_{}_{}".format(env_type, s)
    g_filename = "G_{}_{}".format(env_type, s)
    e_filename = "E_{}_{}".format(env_type, s)
    y_filename = "Y_{}_{}".format(env_type, s)
    
    pickle.dump(X, open(x_filename + ".pkl", "wb"))
    pickle.dump(G, open(g_filename + ".pkl", "wb"))
    pickle.dump(E, open(e_filename + ".pkl", "wb"))
    pickle.dump(Y, open(y_filename + ".pkl", "wb"))
    
    np.save("data/" + x_filename + ".npy", np.vstack(X))
    np.save("data/" + g_filename + ".npy", np.vstack(G))
    np.save("data/" + e_filename + ".npy", np.vstack(E))
    np.save("data/" + y_filename + ".npy", np.hstack(Y))
    
    if platform == "darwin":
        break