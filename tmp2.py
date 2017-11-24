#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:29:12 2017

@author: vitorhadad
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from random import shuffle, choice

from matching.solver.kidney_solver2 import  optimal, compare_optimal
from matching.utils.data_utils import get_additional_regressors, clock_seed
from matching.utils.env_utils import get_actions, snapshot, remove_taken
from matching.utils.data_utils import flatten_matched, disc_mean , get_n_matched
from matching.environment.optn_environment import OPTNKidneyExchange
from matching.tree_search.mcts import *


env = OPTNKidneyExchange(5, .1, 200)

t= 5
#%%
opt = optimal(env, t, t)
ms = set(map(tuple, opt["matched_cycles"][t]))