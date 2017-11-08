#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:44:09 2017

@author: vitorhadad
"""

from itertools import product

#%%
# adv_mcts

ers = [5]
drs = [0.1]
r_horiz = [1, 10]
t_horiz = [10, 100]
scl = [1.41, 5]#, 0.7071, 1]
tpa = [5, 20]
prl = [1, 10]

vs = product(scl,tpa,t_horiz,r_horiz,prl)


for i,args in enumerate(vs):
    cmd = 'qsub -F "{} {} {} {} {}" job.pbs'.format(*args)
    print(cmd)

#%%
# policy_function_traditional
hors = [44]
mcl = [2]
algo = range(44)
sampling = [2] #range(3)

vs = product(hors,mcl,algo,sampling)

for i,args in enumerate(vs):
    cmd = 'qsub -F "{} {} {} {}" job4.pbs'.format(*args)
    print(cmd)
    
    
#%%
# policy_function_gcn
sizes = [1, 5, 10, 20]
num_layers = [1, 2, 5, 10]

vs = product(sizes, num_layers)

for i, args in enumerate(vs):
    cmd = 'qsub -F "{} {}" job5.pbs'.format(*args)
    print(cmd)

#%%%
# Value function gcn
sizes = [1, 5, 10, 20]
num_layers = [1, 2, 5, 10]
horizons = [44]

vs = product(sizes, num_layers, horizons)

for i, args in enumerate(vs):
    cmd = 'qsub -F "{} {} {}" job2.pbs'.format(*args)
    print(cmd)
    

    
    