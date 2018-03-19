from collections import defaultdict
from itertools import chain

import networkx as nx
import numpy as np

from matching.solver import kidney_solver3 as ks
from matching.utils.env_utils import snapshot


def get_priors(actions,
               mu: float = 0,
               sigma: float = 1.001,
               rho: float = 0.5):
    n_actions = len(actions)
    mu = mu * np.ones(n_actions)
    cov = np.zeros((n_actions, n_actions))
    for i, e1 in enumerate(actions):
        for j, e2 in enumerate(actions):
            intersection = np.intersect1d(e1, e2)
            if len(intersection) == 2:
                cov[i, j] = sigma ** 2
            elif len(intersection) == 1:
                cov[i, j] = rho * sigma ** 2
            else:
                cov[i, j] = rho / 2 * sigma ** 2
    return mu, cov


def draw_weights(actions, mu, sigma):
    values = np.random.multivariate_normal(mu, sigma)
    weights = defaultdict(lambda: 1)
    for k, v in enumerate(actions):
        weights[v] = values[k]
    return weights


def compare_solutions(env, t,
                      nodes_to_be_removed,
                      horizon, max_cycle,
                      max_chain, weights):
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
    selected_actions = []
    selected_nodes = []
    for v in solution.getVars():
        if v.x > 0:
            var_name = v.varName
            cycle_or_t, vs = var_name.split("_")
            vs = eval(vs)
            if cycle_or_t == "t":
                vs = vs[:2]
                var_name = "t_" + str(vs)
            selected_actions.append(var_name)
            selected_nodes.append(vs)

    return selected_actions, selected_nodes, solution.ObjVal


def get_reward(env, t, horizon, max_cycle, max_chain, weights):
    current_graph = env.subgraph(env.get_living(t))
    selected_actions, selected_nodes, cur_obj = static_solution(current_graph, max_cycle, max_chain)
    static_solution_nodes = list(chain(*selected_nodes))
    unconstr_reward, constr_reward = compare_solutions(env, t,
                                                       horizon=horizon,
                                                       nodes_to_be_removed=static_solution_nodes,
                                                       max_cycle=max_cycle,
                                                       max_chain=max_chain,
                                                       weights=weights)

    return selected_actions, (constr_reward + cur_obj) / unconstr_reward


def thompson_step(actions, selected, r, mu, sigma):
    action_idx = {action: k for k, action in enumerate(actions)}

    sigma_inv = np.linalg.inv(sigma)

    C = np.full_like(sigma, fill_value=0.25)
    z = np.zeros_like(mu)
    for a1 in selected:
        i = action_idx[a1]
        z[i] = r  # * (r > 0.5)
        for a2 in selected:
            j = action_idx[a2]
            C[i, j] = 0.5  # sigma_inv[i, j]

    post_sigma_inv = sigma_inv + C
    post_sigma = np.linalg.inv(post_sigma_inv)
    post_mu = post_sigma @ (sigma_inv @ mu + C @ z)

    return post_mu, post_sigma


if __name__ == "__main__":

    from matching.environment.optn_environment import OPTNKidneyExchange
    from matching.trimble_solver.interface import greedy

    max_cycle = 2
    max_chain = 2
    horizon = 20
    num_thompson_update = 5
    np.random.seed(12345)

    env = OPTNKidneyExchange(5, 0.1, 100,
                             seed=12345,
                             fraction_ndd=0.05)
    gre = greedy(env=env,
                 max_cycle=max_cycle,
                 max_chain=max_chain)

    opt = ks.solve_with_time_constraints(env,
                                         max_cycle=max_cycle,
                                         max_chain=max_chain)

    this_obj = 0
    matching = dict()
    pool_size = dict()
    for t in range(env.time_length):
        print("\n\n\n\n", t, "\n\n\n\n")
        living = env.get_living(t)
        pool_size[t] = len(living)
        current_graph = nx.subgraph(env, living)
        actions = ks.get_actions(graph=current_graph,
                                 max_cycle=max_cycle,
                                 max_chain=max_chain)
        if len(actions) == 0:
            continue

        mu, sigma = get_priors(actions, mu=0.5)
        sigma += 1e-6 * np.eye(sigma.shape[0])

        for _ in range(num_thompson_update):
            ws = draw_weights(actions, mu, sigma)
            selected, r = get_reward(env, t,
                                     horizon=horizon,
                                     max_cycle=max_cycle,
                                     max_chain=max_chain,
                                     weights=ws)
            mu, sigma = thompson_step(actions, selected, r, mu, sigma)

        print("MU:", mu)
        best_subset = []
        for action_name in np.unique(np.array(actions)[mu > 0.2]):
            ct, vs = action_name.split("_")
            if ct == "c":
                vs = eval(vs)
            else:
                vs = eval(vs)[:2]
            best_subset.extend(vs)

        restricted_graph = env.subgraph(best_subset)
        chosen_actions, chosen_nodes, cur_obj = static_solution(restricted_graph,
                                                                max_cycle,
                                                                max_chain)
        for s in chosen_nodes:
            env.removed_container[t].update(s)

        this_obj += cur_obj

    print(this_obj, gre["obj"], opt.ObjVal)
