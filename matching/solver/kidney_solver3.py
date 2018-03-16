import gurobipy as gb
from collections import defaultdict
import networkx as nx


def get_two_cycles(env, nodes=None):
    if nodes is not None:
        subgraph = env.subgraph(nodes)
    else:
        subgraph = env

    edges = subgraph.edges(data=True)
    output = []
    for u, v, attr in edges:
        if u < v and subgraph.has_edge(v, u):
            output.append((u, v))
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
                output.append((u, w, v))

    return output


def get_cycles(env, max_cycle, nodes=None):
    cycles = []
    if max_cycle >= 2:
        cycles.extend(get_two_cycles(env, nodes))
    if max_cycle >= 3:
        cycles.extend(get_three_cycles(env, nodes))
    return cycles


def shortest_path_to_ndds(graph, i):
    ndds = [n for n, d in graph.nodes(data=True) if d["ndd"]]
    path_lengths = []
    for ndd in ndds:
        try:
            l = nx.shortest_path_length(graph, i, ndd)
            path_lengths.append(l)
        except nx.exception.NetworkXNoPath:
            pass
    return min(path_lengths, default=0)


def solve(graph, max_cycle=2, max_chain=2):
    m = gb.Model()
    m.setParam("Threads", 1)
    m.setParam("OutputFlag", 0)

    ndd_capacity = defaultdict(list)
    incoming_capacity = defaultdict(list)
    sequence_incoming = defaultdict(lambda: defaultdict(list))
    sequence_outgoing = defaultdict(lambda: defaultdict(list))

    grb_vars = []
    for edge in graph.edges():
        v, w = edge
        if graph.node[v]["ndd"]:
            edge_var = m.addVar(vtype=gb.GRB.BINARY, name=str((v, w, 0)))
            grb_vars.append(edge_var)
            ndd_capacity[v].append(edge_var)
            incoming_capacity[w].append(edge_var)
            sequence_incoming[w][0].append(edge_var)
        else:
            for k in range(1, max_chain - 1):
                edge_var = m.addVar(vtype=gb.GRB.BINARY, name=str((v, w, k)))
                grb_vars.append(edge_var)
                incoming_capacity[w].append(edge_var)
                sequence_outgoing[v][k].append(edge_var)
                if k < max_chain - 1:  # ...?
                    sequence_incoming[w][k].append(edge_var)

    m.update()

    # Add constraints
    for i, data in graph.node(data=True):
        if data["ndd"]:
            m.addConstr(gb.quicksum(ndd_capacity[i]) <= 1)
            # print("NDD capacity:", gb.quicksum(ndd_capacity[i]), "<= 1")
        else:
            m.addConstr(gb.quicksum(incoming_capacity[i]) <= 1)
            # print("Pair position capacity:", gb.quicksum(incoming_capacity[i]), "<= 1")

            for k in range(max_chain - 1):
                if sequence_outgoing[i][k + 1]:
                    lhs = gb.quicksum(sequence_incoming[i][k])
                    rhs = gb.quicksum(sequence_outgoing[i][k + 1])
                    m.addConstr(lhs >= rhs)
                    # print("Sequence: ", lhs, ">=", rhs)

    m.update()
    m.setObjective(gb.quicksum(grb_vars), gb.GRB.MAXIMIZE)
    m.optimize()

    return m


def sojourn(graph, i):
    i_entry = graph.node[i]["entry"]
    i_death = graph.node[i]["death"] + 1
    return int(i_entry), int(i_death)


def sojourn_overlap(graph, i, j):
    i_entry = graph.node[i]["entry"]
    i_death = graph.node[i]["death"] + 1
    j_entry = graph.node[j]["entry"]
    j_death = graph.node[j]["death"] + 1

    t_begin = max(i_entry, j_entry)
    t_end = min(i_death, j_death)
    return int(t_begin), int(t_end)


def solve_with_time_constraints(graph,
                                max_cycle,
                                max_chain,
                                weights=None):
    m = gb.Model()
    m.setParam("Threads", 1)
    m.setParam("OutputFlag", 0)
    obj = 0

    # Constraint holders
    capacity = defaultdict(list)
    flow_zero_cur = defaultdict(lambda: defaultdict(list))
    flow_zero_prev = defaultdict(lambda: defaultdict(list))

    flow_nonzero_cur = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    flow_nonzero_prev = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Cycle variables
    if max_cycle >= 2:
        cycles = get_cycles(graph, max_cycle)
        cycle_vars = [m.addVar(vtype=gb.GRB.BINARY, name=str(c)) for c in cycles]
        for cyc, cyc_var in zip(cycles, cycle_vars):
            for v in cyc:
                capacity[v].append(cyc_var)
            if weights is None:
                obj += len(cyc) * cyc_var
            else:
                for i in range(len(cyc)):
                    j = (i+1) % len(cyc)
                    obj += weights[(i, j)] * cyc_var

    # Chain variables
    chain_vars = {}
    if max_chain > 0:
        for edge in graph.edges():
            v, w = edge
            t_begin, t_end = sojourn_overlap(graph, v, w)
            for t in range(t_begin, t_end):
                if graph.node[v]["ndd"]:
                    edge_var = m.addVar(vtype=gb.GRB.BINARY, name=str((v, w, 0, t)))
                    chain_vars[(v, w, 0, t)] = edge_var
                    capacity[v].append(edge_var)
                    capacity[w].append(edge_var)
                else:
                    d = shortest_path_to_ndds(graph, w)
                    for k in range(d, max_chain):
                        edge_var = m.addVar(vtype=gb.GRB.BINARY, name=str((v, w, k, t)))
                        chain_vars[(v, w, k, t)] = edge_var
                        capacity[w].append(edge_var)

        for (v, w, k, t), x in chain_vars.items():
            if k == 0:
                flow_zero_cur[v][t].append(x)
            flow_nonzero_cur[v][t][k].append(x)
            flow_nonzero_prev[w][t][k + 1].append(x)

            t_begin, t_end = sojourn(graph, w)
            for s in range(t, t_end + 1):
                flow_zero_prev[w][s + 1].append(x)

    m.update()

    for q in graph.nodes():

        m.addConstr(gb.quicksum(capacity[q]) <= 1)

        # A vertex that is not NDD...
        if not graph.node[q]["ndd"]:
            for s in range(*sojourn(graph, q)):

                # ...if at position 0, must have received in a previous time, any position
                cur = flow_zero_cur[q][s]
                prev = flow_zero_prev[q][s]
                if len(prev) or len(cur):
                    m.addConstr(gb.quicksum(cur) <= gb.quicksum(prev))

                # If not at position 0, must have received from position k-1 at this time period
                for k in range(max_chain):
                    cur = flow_nonzero_cur[q][s][k]
                    prev = flow_nonzero_prev[q][s][k]
                    if len(prev) or len(cur):
                        m.addConstr(gb.quicksum(cur) <= gb.quicksum(prev))

    m.update()
    if weights is None:
        obj += gb.quicksum(list(chain_vars.values()))
    else:
        obj += gb.quicksum([weights[(i, j)] * v for (i, j, *_), v in chain_vars.items()])

    m.setObjective(obj, gb.GRB.MAXIMIZE)
    m.optimize()

    return m


if __name__ == "__main__":
    import networkx as nx

    from matching.environment.optn_environment import OPTNKidneyExchange

    # for i in range(10):
    env = OPTNKidneyExchange(5, 0.1, 10, seed=0, fraction_ndd=0.1)

    m = solve_with_time_constraints(env, 0, 2)
