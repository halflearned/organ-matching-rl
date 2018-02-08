# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:58:07 2018

@author: halflearned
"""

import numpy as np
from random import choice
from matching.utils.env_utils import two_cycles, snapshot
from matching.utils.data_utils import clock_seed
from matching.solver.kidney_solver2 import optimal

from typing import List
from itertools import combinations, product
from itertools import chain
from scipy.stats import geom


class CombinatorialBandit:

    def __init__(self, env, t: int, gamma: float = .1,
                 iters_per_arm: int = 100,
                 max_match: int = 5):
        self.env = env
        self.t = t
        self.max_match = max_match
        self.cycles = two_cycles(env, t) + [()]
        self.arms = get_arm_matrix(self.cycles, max_match)
        self.n_arms = len(self.arms)

        self.h = max(2, int(geom(env.death_rate).ppf(.9)))
        self.w = np.ones(self.n_arms)
        self.p = np.full_like(self.w, fill_value=1 / self.n_arms)
        self.q = np.full_like(self.w, fill_value=1 / self.n_arms)
        self.mu = np.full_like(self.w, fill_value=1 / self.n_arms)
        self.gamma = gamma
        self.iters_per_arm = iters_per_arm


    def __str__(self):
        return "Combinatorial(gamma={})".format(self.gamma)


    def simulate(self):

        total_iters = self.iters_per_arm * self.n_arms

        for _ in range(total_iters):

            # 1. Update probabilities
            self.p = (1 - self.gamma) * self.q + self.gamma * self.mu

            # 2. Draw action
            a = np.random.choice(self.n_arms, p=self.p)
            v = self.arms[a]

            # 3. Take action, observe rewards
            c = self.get_cost(v)

            # 4. Outer product matrix
            p_matrix = exact_outer_product(self.arms, self.p)
            p_inv = np.linalg.inv(p_matrix)

            # 5. Update pseudo-loss
            ltilde = c * (p_inv @ v[:, np.newaxis])

            # 6. Update weights
            self.q = self.q * np.exp(-self.gamma * self.arms @ ltilde).flatten()
            self.q /= self.q.sum()


    def choose(self):
        best_idx = np.argwhere(self.p == np.max(self.p)).flatten()
        best_arm = self.arms[choice(best_idx)]
        best = list(chain(*[self.cycles[i] for i, x in enumerate(best_arm) if x == 1]))
        return best

    def get_cost(self, v: np.ndarray):
        cycle = list(chain(*[self.cycles[i] for i,x in enumerate(v) if x == 1]))
        snap = snapshot(self.env, self.t)
        snap.removed_container[self.t].update(cycle)
        snap.populate(self.t + 1, self.t + self.h + 1, seed=clock_seed())
        reward = optimal(snap, self.t, self.h + 1)["obj"] + len(cycle)
        return -(reward / self.h)



def get_cycle_combos(cycles: List[tuple],
                     max_match: int) -> List[list]:
    """

    Parameters
    ----------
    max_match
    cycles : List[tuple]

    Returns
    -------

    """

    mapping = {tuple(v): k for k, v in enumerate(cycles)}
    combos = {1: [[c] for c in cycles]}

    if max_match >= 2:
        combos[2] = []
        for c1, c2 in combinations(combos[1], 2):
            combo = c1 + c2
            if len(set(chain(*combo))) == 4:
                combos[2].append(list(combo))

    if max_match >= 3:
        combos[3] = []
        for c1, c2 in product(combos[1], combos[2]):
            combo = c1 + c2
            if len(set(chain(*combo))) == 6:
                combos[3].append(combo)

    if max_match >= 4:
        combos[4] = []
        for c1, c2 in combinations(combos[2], 2):
            combo = c1 + c2
            if len(set(chain(*combo))) == 8:
                combos[4].append(combo)

    if max_match >= 5:
        combos[5] = []
        for c1, c2, c3 in product(combos[1], combos[2], combos[2]):
            combo = c1 + c2 + c3
            if len(set(chain(*combo))) == 10:
                combos[5].append(combo)

        for c1, c2 in product(combos[1], combos[4]):
            combo = c1 + c2
            if len(set(chain(*combo))) == 10:
                combos[5].append(combo)

    if max_match > 5:
        raise NotImplementedError

    cycle_combos = []
    for k,cmbs in combos.items():
        for cmb in cmbs:
            cycle_combos.append([mapping[c] for c in cmb])

    print("Found {} cycle combinations".format(len(cycle_combos)))
    return cycle_combos


def get_arm_matrix(cycles: List[tuple],
                   max_match: int) -> np.ndarray:
    cycle_combos = get_cycle_combos(cycles, max_match)

    V = np.zeros((len(cycle_combos), len(cycles)))
    for i, c in enumerate(cycle_combos):
        V[i, c] = 1

    return V


def exact_outer_product(V: np.ndarray, w: np.ndarray) -> np.ndarray:
    N, d = V.shape
    EVV = np.zeros(shape=(d, d))
    for i in range(N):
        EVV += w[i] * V[i, :, np.newaxis] * V[i, :, np.newaxis].T
    return EVV
