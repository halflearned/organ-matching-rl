import gurobipy as gb
from collections import defaultdict

import networkx as nx
import numpy as np


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


def get_chain_positions(graph, max_chain_length, reduce=False):
    if max_chain_length < 1:
        return []

    ndds = [node for node, data in graph.nodes(data=True) if data["ndd"]]

    variables = []
    for i, j in graph.edges:
        if graph.node[i]["ndd"]:
            variables.append((i, j, 0))
        else:
            if reduce:
                d = min(nx.shortest_path_length(graph, i, ndd) for ndd in ndds)
            else:
                d = 1
            for k in range(d, max_chain_length):  # TODO: Check if d instead of d+1
                variables.append((i, j, k))

    return variables


def solve(graph, max_cycle=2, max_chain=2):

    m = gb.Model()
    m.setParam("Threads", 1)

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
                if k < max_chain - 1: # ...?
                    sequence_incoming[w][k].append(edge_var)

    m.update()

    # Add constraints
    for i, data in graph.node(data=True):
        if data["ndd"]:
            m.addConstr(gb.quicksum(ndd_capacity[i]) <= 1)
            #print("NDD capacity:", gb.quicksum(ndd_capacity[i]), "<= 1")
        else:
            m.addConstr(gb.quicksum(incoming_capacity[i]) <= 1)
            #print("Pair position capacity:", gb.quicksum(incoming_capacity[i]), "<= 1")

            for k in range(max_chain - 1):
                if sequence_outgoing[i][k + 1]:
                    lhs = gb.quicksum(sequence_incoming[i][k])
                    rhs = gb.quicksum(sequence_outgoing[i][k+1])
                    m.addConstr(lhs >= rhs)
                    #print("Sequence: ", lhs, ">=", rhs)

    m.update()
    m.setObjective(gb.quicksum(grb_vars), gb.GRB.MAXIMIZE)
    m.optimize()

    return m


def sojourn(graph, i):
    i_entry = graph.node[i]["entry"]
    i_death = graph.node[i]["death"]
    return i_entry, i_death



def sojourn_overlap(graph, i, j):
    i_entry = graph.node[i]["entry"]
    i_death = graph.node[i]["death"]
    j_entry = graph.node[j]["entry"]
    j_death = graph.node[j]["death"]

    t_begin = max(i_entry, j_entry)
    t_end = min(i_death, j_death)
    return t_begin, t_end+1



def solve_with_time_constraints(graph,
                                max_cycle,
                                max_chain,
                                max_chain_per_period):
    m = gb.Model()
    m.setParam("Threads", 1)

    ndd_capacity = defaultdict(list)
    incoming_capacity = defaultdict(list)
    sequence_incoming = defaultdict(lambda: defaultdict(list))
    sequence_outgoing = defaultdict(lambda: defaultdict(list))

    time_incoming = defaultdict(lambda: defaultdict(list))
    time_outgoing = defaultdict(lambda: defaultdict(list))

    grb_vars = []
    for edge in graph.edges():
        v, w = edge
        for t in sojourn_overlap(graph, v, w):
            if graph.node[v]["ndd"]:
                edge_var = m.addVar(vtype=gb.GRB.BINARY, name=str((v, w, 0, t)))
                grb_vars.append(edge_var)
                ndd_capacity[v].append(edge_var)
                incoming_capacity[w].append(edge_var)
                sequence_incoming[w][0].append(edge_var)

                time_incoming[w][t].append(edge_var)
                time_outgoing[w][t].append(edge_var)
            else:
                for k in range(1, max_chain - 1):
                    edge_var = m.addVar(vtype=gb.GRB.BINARY, name=str((v, w, k, t)))
                    grb_vars.append(edge_var)
                    incoming_capacity[w].append(edge_var)
                    sequence_outgoing[v][k].append(edge_var)
                    if k < max_chain - 1:
                        sequence_incoming[w][k].append(edge_var)

                    time_incoming[w][t].append(edge_var)
                    time_outgoing[w][t].append(edge_var)

    m.update()
    # Add membership and chain length constraints
    for i, data in graph.node(data=True):
        if data["ndd"]:
            m.addConstr(gb.quicksum(ndd_capacity[i]) <= 1)
        else:
            m.addConstr(gb.quicksum(incoming_capacity[i]) <= 1)
            for k in range(max_chain - 1):
                if sequence_outgoing[i][k + 1]:
                    lhs = gb.quicksum(sequence_incoming[i][k])
                    rhs = gb.quicksum(sequence_outgoing[i][k + 1])
                    m.addConstr(lhs >= rhs)

    # Add time constraints
    for i in graph.node:
        for t in sojourn(graph, i):
            if time_outgoing[i][t + 1]:
                lhs = gb.quicksum(time_incoming[i][t])
                rhs = gb.quicksum(time_outgoing[i][t + 1])
                m.addConstr(lhs >= rhs)
                m.addConstr(lhs <= max_chain_per_period)


    m.update()
    m.setObjective(gb.quicksum(grb_vars), gb.GRB.MAXIMIZE)
    m.optimize()

    return m


if __name__ == "__main__":

    import numpy as np
    import networkx as nx
    from matching.trimble_solver.interface import solve as trimble_solve
    from matching.environment.abo_environment import ABOKidneyExchange

    env = ABOKidneyExchange(5, .1, 100, fraction_ndd=0.1)
    m = solve_with_time_constraints(env,
                                      max_cycle=0,
                                      max_chain=3,
                                      max_chain_per_period=1)
    xs = [v for v in m.getVars() if v.x > 0]


