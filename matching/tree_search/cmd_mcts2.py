#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:52:37 2017

@author: vitorhadad
"""

from itertools import product
from os import system
from collections import OrderedDict

config = OrderedDict([('scls', [0.1, 1.4, 5]),
                       ('tpas', [1, 10]),
                       ('prls', [1, 5]),
                       ('t_horizs', [3, 10, 22]),
                       ('r_horizs', [3, 10, 22]),
                       ('gcns', [(10,1), (5,1), (50,1)]),
                       ('deflators', [.85, .9, .95]),
                       ('use_priorss', [True, False])])

for cfg in product(*config.values()):
    cmd = 'qsub -F "\'{}\'" job_mcts.pbs'.format(cfg)
    print(cmd)
    #system(cmd)
                       
