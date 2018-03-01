#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:41:59 2018

@author: vitorhadad
"""

# os.chdir("/Users/vitorhadad/Documents/kidney/matching")

import pickle
from itertools import product

import numpy as np
import pandas as pd
import networkx as nx

from matching.environment.base_environment2 import BaseKidneyExchange


class OPTNKidneyExchange(BaseKidneyExchange):

    optn_pairs = pickle.load(open("matching/optn_data/optn_pairs.pkl", "rb"))

    don_blood_cols = [c for c in optn_pairs.columns if "blood" in c and "don" in c and "cpra" not in c]
    don_tissue_cols = [c for c in optn_pairs.columns if "blood" not in c and "don" in c and "cpra" not in c]
    pat_blood_cols = [c for c in optn_pairs.columns if "blood" in c and "pat" in c and "cpra" not in c]
    pat_tissue_cols = [c for c in optn_pairs.columns if "blood" not in c and "pat" in c and "cpra" not in c]

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
                      populate=False,
                      fraction_ndd=fraction_ndd)

        self.data = None
        if populate: self.populate(seed=seed)

    def __str__(self):
        return "OPTN({},{},{},{})".format(self.entry_rate,
                                          self.death_rate, self.time_length, self.seed)

    # def __class__(self):
    #    return OPTNKidneyExchange

    def initial_populate(self, t_end, seed=None):
        self.data = self.draw_node_features(0, t_end)
        ids = list(self.data.index)
        self.add_nodes_from(ids)

        edges = self.draw_edges(ids, ids)
        self.add_edges_from(edges, weight=1)

    def repopulate(self, t_begin, t_end, seed=None):

        # Get existing ids        
        old_ids = list(self.nodes)
        next_id = max(self.nodes) + 1

        # Draw new data
        new_data = self.draw_node_features(t_begin, t_end)
        new_ids = list(range(next_id, next_id + len(new_data)))
        new_data.index = new_ids

        # Update graph
        self.add_nodes_from(new_ids)

        # Update data
        self.data = pd.concat([self.data, new_data], axis=0)

        # Update edges
        newnew_edges = self.draw_edges(new_ids, new_ids)
        self.add_edges_from(newnew_edges, weight=1)

        oldnew_edges = self.draw_edges(old_ids, new_ids)
        self.add_edges_from(oldnew_edges, weight=1)

        newold_edges = self.draw_edges(new_ids, old_ids)
        self.add_edges_from(newold_edges, weight=1)

    def populate(self, t_begin=0, t_end=None, seed=None):
        np.random.seed(seed)

        if t_end is None:
            t_end = self.time_length

        if self.data is None:
            self.initial_populate(t_end=t_end, seed=seed)
        else:
            self.repopulate(t_begin=t_begin, t_end=t_end, seed=seed)

        # TODO: Refactor
        ndd_status = []
        for ndd in self.data["ndd"]:
            ndd_status.append({"ndd": ndd})
        nx.set_node_attributes(self, dict(zip(self.data.index, ndd_status)))

        entries = []
        for t in self.data["entry"]:
            entries.append({"entry": t})

        nx.set_node_attributes(self, dict(zip(self.data.index, entries)))


    def draw_node_features(self, t_begin, t_end):

        duration = t_end - t_begin
        n_periods = np.random.poisson(self.entry_rate, size=duration)

        n = np.sum(n_periods)
        entries = np.repeat(np.arange(t_begin, t_end), n_periods)

        sojourns = np.random.geometric(self.death_rate, n) - 1
        deaths = entries + sojourns
        ndd = np.random.binomial(n=1, p=self.fraction_ndd, size=n)

        data = self.optn_pairs.sample(n=n)

        with pd.option_context("chained_assignment", None):
            data["entry"] = entries
            data["death"] = deaths
            data["ndd"] = ndd

        return data.reset_index(drop=True)

    def filter_time_compatible(self, source_nodes, target_nodes):
        s_entry = self.data.loc[source_nodes, "entry"].values
        s_death = self.data.loc[source_nodes, "death"].values
        t_entry = self.data.loc[target_nodes, "entry"].values
        t_death = self.data.loc[target_nodes, "death"].values

        comp = (s_entry <= t_death.T) & \
               (s_death >= t_entry.T)

        return source_nodes[comp], target_nodes[comp]

    def filter_blood_compatible(self, source_nodes, target_nodes):

        s_blood = self.data.loc[source_nodes, self.don_blood_cols].values
        t_blood = self.data.loc[target_nodes, self.pat_blood_cols].values

        o_col = self.don_blood_cols.index("blood_O_don")
        ab_col = self.pat_blood_cols.index("blood_AB_pat")

        comp = s_blood[:, o_col] | \
               t_blood[:, ab_col] | \
               np.any(s_blood & t_blood, 1)

        return source_nodes[comp], target_nodes[comp]

    def filter_tissue_compatible(self, source_nodes, target_nodes):

        s_tissue = self.data.loc[source_nodes, self.don_tissue_cols].values
        t_tissue = self.data.loc[target_nodes, self.pat_tissue_cols].values

        comp = np.logical_not(np.any(s_tissue & t_tissue, 1))

        return source_nodes[comp], target_nodes[comp]

    def filter_not_same(self, source_nodes, target_nodes):
        not_same = source_nodes != target_nodes
        return source_nodes[not_same], target_nodes[not_same]

    def draw_edges(self, source_nodes, target_nodes):

        if len(source_nodes) == 0 or len(target_nodes) == 0:
            return []

        not_ndds = ~self.data.loc[target_nodes, "ndd"].astype(bool).values
        target_nodes = np.array(target_nodes)[not_ndds]  # TODO: Refactor

        pairs = np.array(list(product(source_nodes, target_nodes))).T

        pairs = self.filter_blood_compatible(*pairs)

        pairs = self.filter_tissue_compatible(*pairs)

        pairs = self.filter_time_compatible(*pairs)

        pairs = self.filter_not_same(*pairs)

        return np.vstack(pairs).T

    def X(self, t, graph_attributes=True, tissue_dummies=True, dtype="numpy"):

        nodelist = self.get_living(t, indices_only=True)
        n = len(nodelist)
        d = self.data.loc[nodelist, :]
        blood_cols = self.don_blood_cols[1:] + self.pat_blood_cols[1:]

        Xs = np.zeros((n, 11))
        Xs[:, 0] = t - d["entry"]
        Xs[:, 1] = d["cpra_pat"]
        Xs[:, 2] = d["cpra_don"]
        Xs[:, 3:9] = d[blood_cols]
        Xs[:, 10] = d["ndd"]
        # Delete patients info if ndd
        Xs[d["ndd"], 6:9] = 0

        if graph_attributes:
            Xs = np.hstack([Xs,
                            np.full(shape=(n, 1), fill_value=self.entry_rate),
                            np.full(shape=(n, 1), fill_value=self.death_rate)])
        if tissue_dummies:
            tissue_cols = self.don_tissue_cols[1:] + self.pat_tissue_cols[1:]
            Xs = np.hstack([Xs, d[tissue_cols]])

        if dtype == "numpy":
            return Xs

        elif dtype == "pandas":
            columns = ["waiting_time", "patient_cpra", "donor_cpra"] + blood_cols + ["ndd"]
            if graph_attributes:
                columns += ["entry_rate", "death_rate"]
            if tissue_dummies:
                columns += tissue_cols
            return pd.DataFrame(index=nodelist,
                                data=Xs,
                                columns=columns)
        else:
            raise ValueError("Unknown dtype")


# %%
if __name__ == "__main__":

    from matching.utils.data_utils import summary
    from matching.trimble_solver.interface import optimal, greedy

    env = OPTNKidneyExchange(5, .1, 500, fraction_ndd=0.1)
    o = optimal(env, 2, 2)
    g = greedy(env, 2, 2)


