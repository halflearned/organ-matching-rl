#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 12:50:59 2017
z


@author: vitorhadad
"""

import os
from copy import deepcopy
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import trange

from matching.solver.kidney_solver2 import optimal


def summary(env, timing):
    container = deepcopy(env.removed_container)
    pool_sizes = np.zeros(env.time_length)
    match_sizes = np.zeros(env.time_length)

    for t in range(env.time_length):
        matched_t = timing[t]
        pool_sizes[t] = len(env.get_living(t))
        match_sizes[t] = len(matched_t)
        env.removed_container[t].update(matched_t)

    env.removed_container = container
    s = {"match_size": match_sizes, "pool_size": pool_sizes}
    return s


def run_node2vec(A, d=10):
    rnd = clock_seed()
    input = "edges_{}.txt".format(rnd)
    output = "emb_{}.txt".format(rnd)

    G = nx.from_numpy_matrix(A)
    nx.write_edgelist(G, input, data=False)

    cmd = "./node2vec -i:{} -o:{} -d:{} -dr" \
        .format(input, output, d)

    os.system(cmd)

    with open(output, "r") as emb:
        lines = emb.readlines()
        n = len(lines) - 1
        features = np.zeros(shape=(n, d))
        for k, line in enumerate(lines[1:]):
            _, *xs = line.split(" ")
            features[k] = [float(x) for x in xs]

    os.remove(input)
    os.remove(output)

    return features


def get_features(env):
    opt = optimal(env)

    labels = []
    pair_features = []
    networkx_features = []
    node2vec_features = []

    for t in trange(env.time_length):
        try:
            liv = np.array(env.get_living(t))
            A_full = env.A(t)
            has_cycle = np.einsum("ij,ji->i", A_full, A_full) > 0
            if not np.any(has_cycle):
                continue

            X = env.X(t)[has_cycle]
            A = A_full[has_cycle, :][:, has_cycle]
            E = run_node2vec(A)
            G = get_additional_regressors(env, t, dtype="numpy")[has_cycle]

            liv_and_cycle = liv[has_cycle]
            m = opt["matched"][t]

            Y = np.isin(liv_and_cycle, list(m)).astype(int)
            labels.append(Y)

            env.removed_container[t].update(m)

            assert G.shape[0] == E.shape[0] == X.shape[0]
            pair_features.append(X)
            networkx_features.append(G)
            node2vec_features.append(E)

        except Exception as e:
            print(e)
            import pdb;
            pdb.set_trace()

    return pair_features, networkx_features, node2vec_features, labels


def evaluate_policy(net, env, t, dtype="numpy"):
    if "GCN" in str(type(net)):
        X = env.X(t, graph_attributes=True, dtype="numpy")
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])[np.newaxis, :, :]
        A = env.A(t)[np.newaxis, :, :]
        logits, counts = net.forward(A, Z)

    elif "MLP" in str(type(net)):
        X = env.X(t)
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])[np.newaxis, :, :]
        logits, counts = net.forward(Z)

    elif "RNN" in str(type(net)):
        X = env.X(t, graph_attributes=True, dtype="numpy")
        G, N = get_additional_regressors(env, t)
        Z = np.hstack([X, G, N])[np.newaxis, :, :].transpose((1, 0, 2))
        logits, counts = net.forward(Z)
    else:
        raise ValueError("Unknown algorithm")

    probs = F.softmax(logits.squeeze(), dim=1)

    if dtype == "numpy":
        probs = probs[:, 1].data.numpy()
        counts = F.softmax(counts, dim=1)[0, 1].data.numpy()[0]

    return probs, counts


def get_cycle_probabilities(living, cycles, probs):
    liv_idx = {j: i for i, j in enumerate(living)}
    cycle_probs = np.zeros(len(cycles))
    for k, cyc in enumerate(cycles):
        cycle_probs[k] = np.mean([probs[liv_idx[i]] for i in cyc])
    return cycle_probs


def softmax(x, T=1):
    e_x = np.exp((x - np.max(x)) / T)
    return e_x / e_x.sum()


def cumavg(x):
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def clock_seed():
    return int(str(int(time() * 1e8))[10:])


def disc_mean(xs, gamma=0.97):
    return np.mean([gamma ** i * r for i, r in enumerate(xs)])


def balancing_weights(XX, y):
    yy = y.flatten()
    n1 = np.sum(yy)
    n0 = len(yy) - n1
    p = np.zeros(int(n0 + n1))
    p[yy == 0] = 1 / n0
    p[yy == 1] = 1 / n1
    p /= p.sum()
    return p


def get_rewards(solution, t, h):
    return sum([len(match) for period, match in solution["matched"].items()
                if period >= t and period < t + h])


def flatten_matched(m, t_begin=0, t_end=None):
    if t_end is None:
        t_end = np.inf
    matched = []
    for t, x in m.items():
        if t >= t_begin and t <= t_end:
            matched.extend(x)
    return set(matched)


def get_additional_regressors(env, t, dtype="numpy"):
    f = lambda d: list(d.values())

    nodes = env.get_living(t)
    subg = nx.subgraph(env, nodes)

    graph_properties = pd.DataFrame({
        "density": nx.density(subg),
        "number_of_nodes": [subg.number_of_nodes()] * subg.number_of_nodes(),
        "number_of_edges": [subg.number_of_edges()] * subg.number_of_nodes()
    })

    node_properties = {}
    try:
        node_properties["betweenness_centrality"] = f(nx.betweenness_centrality(subg))
    except:
        node_properties["betweenness_centrality"] = [0] * subg.number_of_nodes()

    try:
        node_properties["in_degree_centrality"] = f(nx.in_degree_centrality(subg))
    except:
        node_properties["in_degree_centrality"] = [0] * subg.number_of_nodes()

    try:
        node_properties["out_degree_centrality"] = f(nx.out_degree_centrality(subg))
    except:
        node_properties["out_degree_centrality"] = [0] * subg.number_of_nodes()

    try:
        node_properties["harmonic_centrality"] = f(nx.harmonic_centrality(subg))
    except:
        node_properties["harmonic_centrality"] = [0] * subg.number_of_nodes()

    try:
        node_properties["closeness_centrality"] = f(nx.closeness_centrality(subg))
    except:
        node_properties["closeness_centrality"] = [0] * subg.number_of_nodes()

    node_properties.update({
        "core_number": f(nx.core_number(subg)),
        "pagerank": f(nx.pagerank(subg)),
        "in_edges": [len(subg.in_edges(v)) for v in subg.nodes()],
        "out_edges": [len(subg.out_edges(v)) for v in subg.nodes()],
        "average_neighbor_degree": f(nx.average_neighbor_degree(subg))
    })

    node_properties = pd.DataFrame(node_properties)

    output = pd.concat([graph_properties, node_properties], axis=1)

    if dtype == "pandas":
        return output
    elif dtype == "numpy":
        return output.values


def get_dead(env, matched, t_begin=None, t_end=None):
    if t_begin is None:
        t_begin = 0

    if t_end is None:
        t_end = env.time_length - 1

    would_be_dead = {n for n, d in env.nodes.data()
                     if d["death"] >= t_begin and \
                     d["death"] <= t_end}

    dead = would_be_dead.difference(matched)

    return dead


def pad_and_stack(As, Xs, GNs, Ys):
    n = max(x.shape[0] for x in Xs)
    A_pad = []
    X_pad = []
    GN_pad = []
    Y_pad = []
    for A, X, GN, Y in zip(As, Xs, GNs, Ys):
        r = n - X.shape[0]
        A_pad.append(np.pad(A, ((0, r), (0, r)), mode="constant"))
        X_pad.append(np.pad(X, ((0, r), (0, 0)), mode="constant"))
        GN_pad.append(np.pad(GN, ((0, r), (0, 0)), mode="constant"))
        Y_pad.append(np.pad(Y, ((0, r), (0, 0)), mode="constant"))
    return np.stack(A_pad), np.stack(X_pad), \
           np.stack(GN_pad), np.stack(Y_pad)


def get_deaths(env, solution, t_begin=None, t_end=None):
    if t_begin is None:
        t_begin = 0

    if t_end is None:
        t_end = env.time_length

    R = solution["matched_pairs"]

    deaths = np.zeros(t_end - t_begin)

    for t in range(env.time_length):
        dead = {n for n, d in env.nodes.data()
                if d["death"] == t
                and n not in R}
        deaths[t] = len(dead)

    return deaths


def get_n_matched(matched, t_begin, t_end):
    n_matched = np.zeros(t_end - t_begin)

    for t in range(t_begin, t_end):
        n_matched[t - t_begin] = len(matched[t])

    return n_matched


def stata_to_csv(filepath, outfile, chunksize=10000):
    file = pd.read_stata(filepath,
                         iterator=True,
                         chunksize=chunksize)

    for k, b in enumerate(file):
        if k % 10 == 0:
            print(k)
        b.to_csv(outfile,
                 mode="a",
                 header=k == 0)
