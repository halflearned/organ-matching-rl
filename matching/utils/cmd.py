#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:03:55 2017

@author: vitorhadad
"""

from os import listdir, system
from sys import argv
from numpy import array_split

env_type = "abo"
mcl = 2
chunksize = 10

for entry_rate in range(1, 21):
    infiles = [f for f in listdir("results/{}_data/".format(env_type))
        if f.endswith(".pkl") and 
        f.startswith(env_type + "_{}".format(entry_rate))
        and ("_{}_".format(mcl) in f or "_{}.pkl".format(mcl) in f)]
    chunks = array_split(infiles, len(infiles)/chunksize)
    for i,chunk in enumerate(chunks):
        outfile = "{}_entryrate{}_part{}.pkl".format(env_type, entry_rate, i)
        cmd = "python data_maker.py " + outfile + " ".join(infiles)
        print(cmd)
        system(cmd)


