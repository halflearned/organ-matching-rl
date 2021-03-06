#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:51:19 2017

@author: vitorhadad
"""
import gurobipy as gb
from collections import defaultdict

import networkx as nx
import numpy as np


def get_cycles_and_chains(env,
                          nodes=None,
                          max_cycle_length=2,
                          max_chain_length=0):
    # Cycles
    cycle_weights, cycles = get_cycles(env,
                                       nodes,
                                       max_cycle_length)

    chain_weights, chains = get_chains(env,
                                       nodes,
                                       max_chain_length)

    return cycle_weights + chain_weights, cycles + chains


def find_chains(g, source, depth):
    chains = []
    cur_chain = [source]

    def dls(cur_chain, depth):
        if depth == 0:
            return

        cur_node = cur_chain[-1]
        for n in nx.neighbors(g, cur_node):
            if n not in cur_chain:
                new_chain = cur_chain + [n]
                chains.append(new_chain)
                dls(new_chain, depth - 1)

    dls(cur_chain, depth - 1)
    return chains


def get_chains(env, nodes, max_chain_length=2):
    if max_chain_length < 2:
        return [], []

    if nodes is not None:
        subgraph = env.subgraph(nodes)
    else:
        subgraph = env

    weights = []
    chains = []
    for node, data in subgraph.nodes(data=True):
        if data["ndd"]:
            cs = find_chains(subgraph, node, max_chain_length)
            for c in cs:
                chains.append(set(c))
                weights.append(len(c))

    return weights, chains


def get_cycles(env, nodes, max_cycle_length=2):
    if max_cycle_length < 2:
        return [], []

    if max_cycle_length > 3:
        raise ValueError("Not supported cycle length")

    cycles = []
    c2s = get_two_cycles(env, nodes)
    cycles.extend(c2s)
    weights = [2] * len(c2s)
    if max_cycle_length == 3:
        c3s = get_three_cycles(env, nodes)
        cycles.extend(c3s)
        weights.extend([3] * len(c3s))

    idx = np.random.permutation(len(cycles))
    weights = [weights[i] for i in idx]
    cycles = [cycles[i] for i in idx]

    return weights, cycles


def get_two_cycles(env, nodes=None):
    if nodes is not None:
        subgraph = env.subgraph(nodes)
    else:
        subgraph = env

    edges = subgraph.edges(data=True)
    output = []
    for u, v, attr in edges:
        if u < v and subgraph.has_edge(v, u):
            output.append({u, v})
    return output


def get_three_cycles(env, nodes=None):
    if nodes is not None:
        subgraph = env.subgraph(nodes)
    else:
        subgraph = env

    edges = subgraph.edges()
    nodes = subgraph.nodes()
    output = []
    for u, v in edges:
        for w in nodes:
            if w >= u or w >= v:
                break
            if subgraph.has_edge(v, w) and subgraph.has_edge(w, u):
                output.append({u, w, v})

    return output


def remove_from_cycles(ws_full, cs_full, restrict):
    restrict = set(restrict)
    ws_restr, cs_restr = [], []
    for w, c in zip(ws_full, cs_full):
        if len(c.intersection(restrict)) == 0:
            ws_restr.append(w)
            cs_restr.append(c)
    return ws_restr, cs_restr


def parse_solution(env, cycles, model, t_begin=None, weights=None):
    matched_cycles = defaultdict(list)
    matched = defaultdict(set)
    matched_pairs = set()
    obj = 0
    for k, (xs, cyc) in enumerate(zip(model.getVars(), cycles)):
        if xs.x > 0:

            t = find_matching_date(env, cyc)
            if t_begin is not None:
                t = max(t, t_begin)

            matched[t].update(cyc)
            matched_cycles[t].append(cyc)
            for v in cyc:
                matched_pairs.add(v)
            if weights is None:
                obj += len(cyc)
            else:
                obj += weights[k]

    return {"matched": matched,
            "matched_pairs": matched_pairs,
            "matched_cycles": matched_cycles,
            "obj": obj}


def solve(weights, cycles):
    cycle_constraints = defaultdict(list)

    m = gb.Model()
    m.setParam("OutputFlag", 0)
    m.setParam("Threads", 1)

    xs = [m.addVar(vtype=gb.GRB.BINARY) for _ in cycles]

    for x, cyc in zip(xs, cycles):
        for v in cyc:
            cycle_constraints[v].append(x)

    for v in cycle_constraints:
        m.addConstr(gb.quicksum(cycle_constraints[v]) <= 1)

    m.update()

    m.setObjective(gb.quicksum([w * v for w, v in zip(weights, xs)]),
                   gb.GRB.MAXIMIZE)
    m.optimize()
    return m


def find_matching_date(env, nodes):
    try:
        return max(env.data.loc[nodes, "entry"])
    except AttributeError:
        return max(env.node[v]["entry"] for v in nodes)


def optimal_with_discount(env,
                          t_begin=None, t_end=None,
                          max_cycle_length=2,
                          gamma=.99):
    if t_begin is None:
        t_begin = 0
    if t_end is None:
        t_end = env.time_length

    nodes = set(env.get_living(t_begin, t_end))
    ws, cs = get_cycles(env, nodes, max_cycle_length)
    for k, (w, c) in enumerate(zip(ws, cs)):
        t = find_matching_date(env, c)
        ws[k] = gamma ** (t - t_begin)

    m = solve(ws, cs)
    return parse_solution(env, cs, m, t_begin, weights=ws)


def optimal(env,
            t_begin=None,
            t_end=None,
            subset=None,
            max_cycle_length=2,
            max_chain_length=0):
    if t_begin is None:
        t_begin = 0
    if t_end is None:
        t_end = env.time_length

    nodes = set(env.get_living(t_begin, t_end))

    if subset is not None:
        nodes = nodes.intersection(subset)

    ws, cs = get_cycles_and_chains(env,
                                   nodes=nodes,
                                   max_cycle_length=max_cycle_length,
                                   max_chain_length=max_chain_length)

    m = solve(ws, cs)

    solution = parse_solution(env, cs, m, t_begin)

    return solution


def compare_optimal(env,
                    t_begin,
                    t_end,
                    perturb,
                    max_cycle_length=2):
    if t_begin is None:
        t_begin = 0
    if t_end is None:
        t_end = env.time_length

    if not isinstance(perturb, set):
        perturb = set(perturb)

    nodes = set(env.get_living(t_begin, t_end))

    ws_full, cs_full = get_cycles(env, nodes, max_cycle_length)

    ws_take, cs_take = remove_from_cycles(ws_full, cs_full, perturb)

    i = cs_full.index(perturb)
    ws_leave = ws_full[:i] + ws_full[i + 1:]
    cs_leave = cs_full[:i] + cs_full[i + 1:]

    m_take = solve(ws_take, cs_take)
    m_leave = solve(ws_leave, cs_leave)

    sol_take = parse_solution(env, cs_take, m_take, t_begin)
    sol_leave = parse_solution(env, cs_leave, m_leave, t_begin)

    sol_take["obj"] += len(perturb)
    sol_take["matched"][t_begin].update(perturb)
    sol_take["matched_cycles"][t_begin].append(perturb)

    return sol_take["obj"], sol_leave["obj"]


def same_rewards(env,
                 t_begin,
                 t_end,
                 perturb,
                 max_cycle_length=2):
    if t_begin is None:
        t_begin = 0
    if t_end is None:
        t_end = env.time_length

    if not isinstance(perturb, set):
        perturb = set(perturb)

    nodes = set(env.get_living(t_begin, t_end))

    ws_leave, cs_leave = get_cycles(env, nodes, max_cycle_length)
    ws_take, cs_take = remove_from_cycles(ws_leave, cs_leave, perturb)

    m_take = solve(ws_take, cs_take)
    m_leave = solve(ws_leave, cs_leave)

    sol_take = parse_solution(env, cs_take, m_take, t_begin)
    sol_leave = parse_solution(env, cs_leave, m_leave, t_begin)

    sol_take["obj"] += len(perturb)
    sol_take["matched"][t_begin].update(perturb)
    sol_take["matched_cycles"][t_begin].append(perturb)

    return sol_take["obj"] == sol_leave["obj"]


def greedy(env, t_begin=None, t_end=None, max_cycle_length=2):
    if t_begin is None:
        t_begin = 0
    if t_end is None:
        t_end = env.time_length

    removed = set()
    matched = defaultdict(set)
    obj = 0
    for t in range(t_begin, t_end):
        nodes = set(env.get_living(t))
        ws, cs = get_cycles(env, nodes - removed, max_cycle_length)
        if not cs:
            continue
        solution = solve(ws, cs)
        m = parse_solution(env, cs, solution, t)["matched_pairs"]
        removed |= m
        matched[t] = m
        obj += len(m)

    return {"matched": matched,
            "matched_pairs": removed,
            "obj": obj}
