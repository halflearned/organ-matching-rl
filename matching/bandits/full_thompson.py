from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
import networkx as nx

from matching.environment.abo_environment import ABOKidneyExchange
from matching.solver import kidney_solver3 as ks
from matching.utils.env_utils import snapshot


def get_priors(graph,
               mu: float = 0,
               sigma: float = 1.0,
               rho: float = 0.5):
    edges = list(graph.edges())
    n_edges = graph.number_of_edges()
    mu = mu * np.ones(n_edges)
    cov = np.eye(n_edges) + np.ones((n_edges, n_edges))
    return mu, cov


def draw_weights(graph, mu, sigma):
    values = np.random.multivariate_normal(mu, sigma)
    weights = defaultdict(lambda: 1)
    for k, (i, j) in enumerate(graph.edges()):
        weights[(i, j)] = values[k]
    return weights


def compare_solutions(env, t, nodes_to_be_removed, horizon,
                      max_cycle, max_chain, weights):
    # Unconstrained
    graph = snapshot(env, t)
    graph.populate(t + 1, t + horizon + 1)
    unconstr_solution = ks.solve_with_time_constraints(graph=graph,
                                                       max_cycle=max_cycle,
                                                       max_chain=max_chain,
                                                       weights=weights)

    # Constrained
    graph.remove_nodes_from(nodes_to_be_removed)
    constr_solution = ks.solve_with_time_constraints(graph=graph,
                                                     max_cycle=max_cycle,
                                                     max_chain=max_chain,
                                                     weights=weights)

    return unconstr_solution.ObjVal, constr_solution.ObjVal


def static_solution(graph, max_cycle, max_chain):
    solution = ks.solve(graph, max_cycle=max_cycle, max_chain=max_chain)
    selected = []
    for v in solution.getVars():
        if v.x > 0:
            tup = eval(v.varName)
            # TODO: Redo this allowing for max_cycle = 4
            if len(tup) < 4:
                selected.append(tup[:2])
            else:
                selected.append(tup)

    return selected, solution.ObjVal


def get_reward(env, t, horizon, max_cycle, max_chain, weights):
    current_graph = env.subgraph(env.get_living(t))
    selected, cur_obj = static_solution(current_graph, max_cycle, max_chain)
    static_solution_nodes = list(chain(*selected))
    unconstr_reward, constr_reward = compare_solutions(env, t, horizon=horizon,
                                                       nodes_to_be_removed=static_solution_nodes,
                                                       max_cycle=max_cycle,
                                                       max_chain=max_chain,
                                                       weights=weights)

    return selected, (constr_reward + cur_obj) / unconstr_reward


def thompson_step(current_graph, selected, r, mu, sigma):
    edges = current_graph.edges
    edge_idx = {edge: k for k, edge in enumerate(edges)}

    sigma_inv = np.linalg.inv(sigma)

    C = np.zeros_like(sigma)
    z = np.zeros_like(mu)
    for edge1 in selected:
        i = edge_idx[edge1]
        z[i] = r * (r > 0.5)
        for edge2 in selected:
            j = edge_idx[edge2]
            C[i, j] = sigma_inv[i, j]

    post_sigma_inv = sigma_inv + C
    post_sigma = np.linalg.inv(post_sigma_inv)
    post_mu = post_sigma @ (sigma_inv @ mu + C @ z)

    return post_mu, post_sigma


if __name__ == "__main__":
    max_cycle = 0
    max_chain = 2
    env = ABOKidneyExchange(5, 0.1, 10, seed=1234, fraction_ndd=0.1)
    t = 8
    horizon = 20
    current_graph = env.subgraph(env.get_living(t))
    mu, sigma = get_priors(current_graph, sigma=1)

    for _ in range(10):
        ws = draw_weights(current_graph, mu, sigma)
        selected, r = get_reward(env, t, horizon, max_cycle, max_chain, ws)
        mu, sigma = thompson_step(current_graph, selected, r, mu, sigma)
