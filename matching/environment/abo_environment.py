#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:05:52 2017

@author: vitorhadad
"""

import numpy as np
import pandas as pd

from matching.environment.base_environment import BaseKidneyExchange


class ABOKidneyExchange(BaseKidneyExchange):
    # Computed using pc = 0.11
    blood_types = np.vstack([(0, 0), (0, 1), (0, 2), (0, 3),
                             (1, 0), (1, 1), (1, 2), (1, 3),
                             (2, 0), (2, 1), (2, 2), (2, 3),
                             (3, 0), (3, 1), (3, 2), (3, 3)])

    blood_prob = np.array([0.05868620827131898, 0.37381232862325586,
                           0.1582579321891519, 0.04266757975687919,
                           0.04111935614855814, 0.02881088248630798,
                           0.11088575099169286, 0.029895668159525032,
                           0.01740837254080671, 0.11088575099169286,
                           0.005163929370226835, 0.01265668963290891,
                           0.004693433773256711, 0.0032885234975477537,
                           0.0013922358596199801, 0.0003753577072504848])

    def __init__(self,
                 entry_rate,
                 death_rate,
                 time_length,
                 seed=None,
                 populate=True,
                 fraction_ndd=0):

        super(ABOKidneyExchange, self) \
            .__init__(entry_rate=entry_rate, death_rate=death_rate,
                      time_length=time_length, seed=seed,
                      populate=populate, fraction_ndd=fraction_ndd)

    def __str__(self):
        return "ABO({},{},{},{},{})".format(self.entry_rate,
                                            self.death_rate,
                                            self.time_length,
                                            self.fraction_ndd,
                                            self.seed)

    def draw_node_features(self, t_begin, t_end):

        if t_begin == 0:
            np.random.seed(self.seed)

        duration = t_end - t_begin
        n_periods = np.random.poisson(self.entry_rate, size=duration)

        n = np.sum(n_periods)
        entries = np.repeat(np.arange(t_begin, t_end), n_periods).reshape(-1, 1)

        sojourns = np.random.geometric(p=self.death_rate, size=(n, 1)) - 1
        deaths = entries + sojourns
        ndd = np.random.binomial(n=1, p=self.fraction_ndd, size=(n, 1))

        idx = np.random.choice(len(self.blood_types), p=self.blood_prob, size=n)

        blood = self.blood_types[idx]

        data = np.hstack([entries, deaths, ndd, blood])

        colnames = ["entry", "death", "ndd", "p_blood", "d_blood"]
        results = []
        for row in data:
            results.append(dict(zip(colnames, row)))

        return results

    def draw_edges(self, source_nodes, target_nodes):

        np.random.seed(self.seed)
        source_nodes = np.array(source_nodes)
        target_nodes = np.array(target_nodes)

        # Block ndds from receiving
        ndds = self.attr("ndd", nodes=target_nodes).astype(bool).flatten()
        target_nodes = target_nodes[~ndds]

        source_entry = self.attr("entry", nodes=source_nodes)
        source_death = self.attr("death", nodes=source_nodes)
        source_don = self.attr("d_blood", nodes=source_nodes)

        target_entry = self.attr("entry", nodes=target_nodes)
        target_death = self.attr("death", nodes=target_nodes)
        target_pat = self.attr("p_blood", nodes=target_nodes)

        time_comp = (source_entry <= target_death.T) & (source_death >= target_entry.T)
        blood_comp = (source_don == target_pat.T) | (source_don == 0) | (target_pat.T == 3)
        not_same = source_nodes.reshape(-1, 1) != target_nodes.reshape(1, -1)

        comp = time_comp & blood_comp & not_same

        s_idx, t_idx = np.argwhere(comp).T

        return list(zip(source_nodes[s_idx], target_nodes[t_idx]))

    def X(self, t, graph_attributes=True, dtype="numpy"):

        nodelist = self.get_living(t, indices_only=False)
        n = len(nodelist)
        Xs = np.zeros((n, 9 + 2 * graph_attributes))
        indices = []
        for i, (n, d) in enumerate(nodelist):
            Xs[i, 0] = (d["p_blood"] == 0) & ~d["ndd"]
            Xs[i, 1] = d["p_blood"] == 1 & ~d["ndd"]
            Xs[i, 2] = d["p_blood"] == 2 & ~d["ndd"]
            Xs[i, 3] = d["d_blood"] == 0
            Xs[i, 4] = d["d_blood"] == 1
            Xs[i, 5] = d["d_blood"] == 2
            Xs[i, 6] = t - d["entry"]
            Xs[i, 7] = d["death"] - t
            Xs[i, 8] = d["ndd"]
            if graph_attributes:
                Xs[i, 9] = self.entry_rate
                Xs[i, 10] = self.death_rate
            indices.append(n)

        if dtype == "numpy":
            return Xs

        elif dtype == "pandas":
            columns = ["pO", "pA", "pAB",
                       "dO", "dA", "dB",
                       "waiting_time",
                       "time_to_death",
                       "ndd"]
            if graph_attributes:
                columns += ["entry_rate", "death_rate"]
            return pd.DataFrame(index=indices,
                                data=Xs,
                                columns=columns)
        else:
            raise ValueError("Unknown dtype")


# %%
if __name__ == "__main__":

    from matching.solver.kidney_solver2 import optimal

    env = ABOKidneyExchange(entry_rate=5, death_rate=0.1,
                            time_length=1000,
                            fraction_ndd=0.2,
                            seed=12345)

    A, X = env.A(3), env.X(3)
    t = 3

    opt = optimal(env,
                  max_cycle_length=2,
                  max_chain_length=3)




