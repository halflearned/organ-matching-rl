#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:24:21 2017

@author: vitorhadad
"""

from matching.solver.kidney_solver2 import optimal

t = 1000
n_iter = 100
ms = [optimal(env, t, t)["matched_pairs"] for _ in range(n_iter)]

ms2 = list(chain(*[list(m) for m in ms]))

c = Counter(ms2)
for i,n in c.items():
    c[i] = n/n_iter