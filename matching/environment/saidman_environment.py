# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:05:52 2017

@author: vitorhadad
"""

from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd

from matching.environment.base_environment import BaseKidneyExchange


class SaidmanKidneyExchange(BaseKidneyExchange):
    blood_types = np.vstack([(0, 0), (0, 1), (0, 2), (0, 3),
                             (1, 0), (1, 1), (1, 2), (1, 3),
                             (2, 0), (2, 1), (2, 2), (2, 3),
                             (3, 0), (3, 1), (3, 2), (3, 3)])

    usa_blood_freq = OrderedDict([(0, 0.44),  # O
                                  (1, 0.42),  # A
                                  (2, 0.1),  # B
                                  (3, 0.04)])  # AB

    # Blood type frequencies
    pc = 0.11
    pc_spouse = 0.33
    blood_prob = OrderedDict()
    blood_prob_spouse = OrderedDict()
    for (x, px), (y, py) in product(*[usa_blood_freq.items()] * 2):
        if x == 'ab' or y == 'o' or x == y:
            blood_prob[(x, y)] = pc * px * py
            blood_prob_spouse[(x, y)] = pc_spouse * px * py
        else:
            blood_prob[(x, y)] = px * py
            blood_prob_spouse[(x, y)] = px * py
    s = sum(blood_prob.values())
    s_sp = sum(blood_prob_spouse.values())
    for (x, y) in blood_types:
        blood_prob[(x, y)] /= s
        blood_prob_spouse[(x, y)] /= s_sp

    # PRA: Low, Medium, High and their crossmatch probs
    pra_freq = np.array([0.7019, 0.2, 0.0981])
    crossmatch_prob = np.array([0.05, 0.45, 0.9])

    # Gender and spouse
    patient_is_female_freq = 0.4090
    donor_is_spouse_freq = 0.4897

    def __init__(self,
                 entry_rate,
                 death_rate,
                 time_length,
                 seed=None,
                 populate=True,
                 fraction_ndd=0):

        super(self.__class__, self) \
            .__init__(entry_rate=entry_rate,
                      death_rate=death_rate,
                      time_length=time_length,
                      seed=seed,
                      populate=populate,
                      fraction_ndd=fraction_ndd)

    def __str__(self):
        return "RSU({},{},{},{})".format(self.entry_rate,
                                         self.death_rate,
                                         self.time_length,
                                         self.seed)

    def draw_blood_type(self, n):
        pat_is_female = np.random.uniform(size=n) < self.patient_is_female_freq
        don_is_husband = np.random.uniform(size=n) < self.donor_is_spouse_freq
        n_spouses = np.sum(pat_is_female & don_is_husband)

        idx = np.random.choice(len(self.blood_types),
                               p=list(self.blood_prob.values()),
                               size=n - n_spouses)

        idx_spouse = np.random.choice(len(self.blood_types),
                                      p=list(self.blood_prob_spouse.values()),
                                      size=n_spouses)

        blood = np.vstack([self.blood_types[idx],
                           self.blood_types[idx_spouse]])

        np.random.shuffle(blood)
        return blood

    def draw_node_features(self, t_begin, t_end):

        if t_begin == 0:
            np.random.seed(self.seed)

        duration = t_end - t_begin
        n_periods = np.random.poisson(self.entry_rate, size=duration)
        n = np.sum(n_periods)

        entries = np.repeat(np.arange(t_begin, t_end), n_periods).reshape(-1, 1)
        sojourns = np.random.geometric(self.death_rate, (n, 1)) - 1
        deaths = entries + sojourns
        ndd = np.random.binomial(n=1, p=self.fraction_ndd, size=(n, 1))

        blood = self.draw_blood_type(n)

        pra = np.random.choice(self.crossmatch_prob,
                               p=self.pra_freq,
                               size=(n, 1))

        data = np.hstack([entries, deaths, blood, ndd, pra])

        colnames = ["entry", "death", "p_blood", "d_blood", "ndd", "pra"]

        results = []
        for row in data:
            results.append(dict(zip(colnames, row)))

        return results

    def draw_edges(self, source_nodes, target_nodes):

        np.random.seed(self.seed)
        source_nodes = np.array(source_nodes)
        target_nodes = np.array(target_nodes)

        # Block ndds from being recipients
        ndds = self.attr("ndd", nodes=target_nodes).astype(bool).flatten()
        target_nodes = target_nodes[~ndds]

        ns = len(source_nodes)
        nt = len(target_nodes)

        source_entry = self.attr("entry", nodes=source_nodes)
        source_death = self.attr("death", nodes=source_nodes)
        source_don = self.attr("d_blood", nodes=source_nodes)
        source_pra = self.attr("pra", nodes=source_nodes)

        target_entry = self.attr("entry", nodes=target_nodes)
        target_death = self.attr("death", nodes=target_nodes)
        target_pat = self.attr("p_blood", nodes=target_nodes)

        hist_comp = np.random.uniform(size=(ns, nt)) > source_pra
        time_comp = (source_entry <= target_death.T) & (source_death >= target_entry.T)
        blood_comp = (source_don == target_pat.T) | (source_don == 0) | (target_pat.T == 3)
        not_same = source_nodes.reshape(-1, 1) != target_nodes.reshape(1, -1)

        comp = hist_comp & time_comp & blood_comp & not_same

        s_idx, t_idx = np.argwhere(comp).T

        return list(zip(source_nodes[s_idx], target_nodes[t_idx]))

    def X(self, t, graph_attributes=True, dtype="numpy"):

        nodelist = self.get_living(t, indices_only=False)
        n = len(nodelist)
        Xs = np.zeros((n, 9 + 2 * graph_attributes))
        indices = []
        for i, (n, d) in enumerate(nodelist):
            Xs[i, 0] = (d["p_blood"] == 0) & (d["ndd"] == 0)
            Xs[i, 1] = (d["p_blood"] == 1) & (d["ndd"] == 0)
            Xs[i, 2] = (d["p_blood"] == 2) & (d["ndd"] == 0)
            Xs[i, 3] = d["d_blood"] == 0
            Xs[i, 4] = d["d_blood"] == 1
            Xs[i, 5] = d["d_blood"] == 2
            Xs[i, 6] = t - d["entry"]
            Xs[i, 7] = d["pra"]
            Xs[i, 8] = d["ndd"]
            if graph_attributes:
                Xs[i, 9] = self.entry_rate
                Xs[i, 10] = self.death_rate
            indices.append(n)

        if dtype == "pandas":
            columns = ["pO", "pA", "pB", "dO", "dA", "dB", "waiting_time", "pra", "ndd"]
            if graph_attributes:
                columns += ["entry_rate", "death_rate"]
            return pd.DataFrame(index=indices, data=Xs, columns=columns)

        elif dtype == "numpy":
            return Xs
        else:
            raise ValueError("Invalid dtype")



if __name__ == "__main__":
    env = SaidmanKidneyExchange(entry_rate=5,
                                death_rate=0.1,
                                time_length=100,
                                fraction_ndd=0.1)

    A, X = env.A(3), env.X(3)


