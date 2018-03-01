from collections import defaultdict
from copy import deepcopy

import matching.trimble_solver.kidney_ip as k_ip
from matching.trimble_solver.kidney_digraph import Digraph
from matching.trimble_solver.kidney_ndds import Ndd, NddEdge


def separate_ndds(g):
    g_pairs, g_ndds = [], []
    for node, data in g.nodes(data=True):
        if data["ndd"]:
            g_ndds.append(node)
        else:
            g_pairs.append(node)
    return g_pairs, g_ndds


def nx_to_trimble(g):
    """Reads a digraph from a networkx DiGraph into the timble input format."""

    g_pairs, g_ndds = separate_ndds(g)
    digraph = Digraph(len(g_pairs))
    ndd_map = {v: k for k, v in enumerate(g_ndds)}
    pair_map = {v: k for k, v in enumerate(g_pairs)}
    ndds = [Ndd() for _ in g_ndds]

    for v, w in g.edges():
        if not g.node[v]["ndd"]:
            src_id, tgt_id = pair_map[v], pair_map[w]
            digraph.add_edge(1, digraph.vs[src_id], digraph.vs[tgt_id])
        else:
            src_id, tgt_id = ndd_map[v], pair_map[w]
            ndds[src_id].add_edge(NddEdge(digraph.vs[tgt_id], 1))

    return digraph, ndds


def solve(g, max_cycle, max_chain, formulation="hpief_prime_full_red"):
    if formulation == "hpief_prime_full_red":
        fn = k_ip.optimise_hpief_prime_full_red
    elif formulation == "hpief_prime":
        fn = k_ip.optimise_hpief_prime
    elif formulation == "hpief_2prime":
        fn = k_ip.optimise_hpief_2prime
    elif formulation == "picef":
        fn = k_ip.optimise_picef
    elif formulation == "ccf":
        fn = k_ip.optimise_ccf
    else:
        raise ValueError("Cannot understand formulation")

    d, ndds = nx_to_trimble(g)
    opt_result = fn(k_ip.OptConfig(d, ndds, max_cycle, max_chain))
    return opt_result


def get_chain_match_date(env, chain):
    match_dates = defaultdict(list)
    t = 0
    for i in range(1, len(chain)):
        v, w = chain[i - 1], chain[i]
        t = max(get_max_entry(env, [v, w]), t)
        if i == 1:
            match_dates[t].extend([v, w])
        else:
            match_dates[t].append(w)
    return match_dates


def get_max_entry(env, nodes):
    try:
        return max(env.data.loc[nodes, "entry"])
    except AttributeError:
        return max(env.node[v]["entry"] for v in nodes)


def parse_trimble_solution(opt, g):
    g_pairs, g_ndds = separate_ndds(g)
    matched = []
    timing = defaultdict(list)
    for cyc in opt.cycles:
        vs = []
        for v in cyc:
            vs.append(g_pairs[v.id])

        matched.extend(vs)
        t = get_max_entry(g, vs)
        timing[t].extend(vs)

    for chain in opt.chains:
        head, rest = chain.ndd_index, chain.vtx_indices
        vs = [g_ndds[head]]
        for v in rest:
            vs.append(g_pairs[v])
        matched.extend(vs)
        dates = get_chain_match_date(g, vs)
        for t, vs in dates.items():
            timing[t].extend(vs)

    return opt.ip_model.ObjVal, matched, timing




def optimal(env, max_cycle, max_chain, t_begin=None, t_end=None):
    if t_begin is None:
        t_begin = 0
    if t_end is None:
        t_end = env.time_length

    g = env.subgraph(env.get_living(t_begin, t_end))
    opt = solve(g, max_cycle=max_cycle, max_chain=max_chain)
    obj, matched, timing = parse_trimble_solution(opt, g)
    return {"obj": obj,
            "matched": matched,
            "timing": timing}


def greedy(env, max_cycle, max_chain, t_begin=None, t_end=None):
    if t_begin is None:
        t_begin = 0
    if t_end is None:
        t_end = env.time_length

    container = deepcopy(env.removed_container)
    obj = 0
    matched = []
    timing = defaultdict(list)
    for t in range(t_begin, t_end):
        opt_t = optimal(env, max_cycle, max_chain, t_begin=t, t_end=t)
        obj += opt_t["obj"]
        matched.extend(opt_t["matched"])
        env.removed_container[t].update(opt_t["matched"])
        timing[t].extend(opt_t["matched"])

    env.removed_container = container
    return {"obj": obj, "matched": matched, "timing": timing}


if __name__ == "__main__":
    from matching.environment.abo_environment import ABOKidneyExchange

    env = ABOKidneyExchange(entry_rate=5,
                            death_rate=0.1,
                            time_length=100,
                            fraction_ndd=0.1)

    g = greedy(env, 2, 2)
    o = optimal(env, 2, 2)
