#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:46:23 2017

@author: vitorhadad
"""

import pickle
import numpy as np
from sys import argv
import re
from tqdm import trange

import matching.utils.data_utils as utils
from sys import argv

size = int(argv[1])
outfile_suffix = argv[2]
infiles = argv[3:]

As = []; Xs = []; GNs = []; Ys = [];
for filename in infiles:
    file = pickle.load(open(filename, "rb"))
    A,X,GN,Y = utils.get_regressors(file, size)
    As.extend(A)
    Xs.extend(X)
    GNs.extend(GN)
    Ys.extend(Y)  
 
Apad, Xpad, GNpad, Ypad = utils.pad_and_stack(As, Xs, GNs, Ys)
np.save("A_{}.npy".format(outfile_suffix), Apad)
np.save("X_{}.npy".format(outfile_suffix), Xpad)
np.save("GN_{}.npy".format(outfile_suffix), GNpad)
np.save("Y_{}.npy".format(outfile_suffix), Ypad)
    



        
        
        
        


