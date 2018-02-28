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


def solve(graph, max_cycle_length=2, max_chain_length=2, name_gbvars=False):
    two_cycles = get_two_cycles(graph)
    if max_cycle_length > 2:
        three_cycles = get_three_cycles(graph)
        cycles = two_cycles + three_cycles
    else:
        cycles = two_cycles

    chain_pos = get_chain_positions(graph, max_chain_length=max_chain_length)

    m = gb.Model()
    # m.setParam("OutputFlag", 0)
    m.setParam("Threads", 1)

    if name_gbvars:
        cycle_gbvars = [m.addVar(vtype=gb.GRB.BINARY, name=str(x)) for x in cycles]
        chain_gbvars = [m.addVar(vtype=gb.GRB.BINARY, name=str(x)) for x in chain_pos]
    else:
        cycle_gbvars = [m.addVar(vtype=gb.GRB.BINARY) for _ in cycles]
        chain_gbvars = [m.addVar(vtype=gb.GRB.BINARY) for _ in chain_pos]

    ndd_capacity_constraints = defaultdict(list)
    pair_position_constraints = defaultdict(list)
    sequence_constraints_lhs = defaultdict(list)
    sequence_constraints_rhs = defaultdict(list)

    # Chain terms
    for (i, j, k), cx in zip(chain_pos, chain_gbvars):
        if graph.node[i]["ndd"]:
            # Ndd-capacity constraints (Equation 4b)
            ndd_capacity_constraints[i].append(cx)
        #else:
            # Position constraints (Equations 4a, first term)
            # Sequence constraints (Equation 4c)
        sequence_constraints_lhs[(j, k)].append(cx)
        sequence_constraints_rhs[(i, k)].append(cx)

        pair_position_constraints[j].append(cx)

    # Cycle terms (Equation 4a, second term)
    for cyc, cx in zip(cycles, cycle_gbvars):
        for i in cyc:
            pair_position_constraints[i].append(cx)

    m.update()
    #import pdb; pdb.set_trace()

    # Add constraints
    for i, data in graph.node(data=True):
        if data["ndd"]:
            # Ndd-capacity constraints (Equation 4b)
            m.addConstr(gb.quicksum(ndd_capacity_constraints[i]) <= 1)
            print("NDD capacity:", gb.quicksum(ndd_capacity_constraints[i]), "<= 1")
        else:
            # Position constraints (Equations 4a)
            m.addConstr(gb.quicksum(pair_position_constraints[i]) <= 1)
            print("Pair position capacity:", gb.quicksum(pair_position_constraints[i]), "<= 1")

        for k in range(max_chain_length-1):
            # Sequence constraints (Equation 4c) If something's wrong, it's here...
            #if (i, k) in sequence_constraints_lhs and (i, k + 1) in sequence_constraints_rhs:
            if len(sequence_constraints_lhs[(i, k)]):
                lhs = gb.quicksum(sequence_constraints_lhs[(i, k)])
                rhs = gb.quicksum(sequence_constraints_rhs[(i, k + 1)])
                m.addConstr(lhs >= rhs)
                print("Sequence: ", lhs, ">=", rhs)

    m.update()
    m.setObjective(gb.quicksum(cycle_gbvars + chain_gbvars), gb.GRB.MAXIMIZE)
    m.optimize()

#    import pdb; pdb.set_trace()

    return m


if __name__ == "__main__":

    from matching.trimble_solver.kidney_digraph import read_digraph
    import matching.trimble_solver.kidney_ip as k_ip
    import matching.trimble_solver.kidney_ndds as k_ndds
    import matching.trimble_solver.kidney_utils as k_utils

    import pandas as pd
    path = "/Users/vitorhadad/Documents/kidney/matching/matching/"
    pairs = pd.read_table(path + "trimble_solver/100-random-weights.input",
                          skiprows=1, header=None, engine='python', skipfooter=1)
    ndds = pd.read_table(path + "trimble_solver/100-random-weights.ndds",
                         skiprows=1, header=None, engine='python', skipfooter=1)


    ndds[0] = ndds[0] + max(pairs[0]) + 1
    pairs_list = pairs[[0, 1]].apply(tuple, axis=1).tolist()
    ndds_list = ndds[[0, 1]].apply(tuple, axis=1).tolist()

    g = nx.DiGraph()
    g.add_nodes_from(range(max(ndds[0])))
    g.add_edges_from(pairs_list)
    g.add_edges_from(ndds_list)
    nx.set_node_attributes(g, name="ndd", values={k:k>=100 for k in range(max(ndds[0]+1))})


    m = solve(g, 2, 2, True)


#    pass
    # Move these to tests later
    # graph = nx.DiGraph()
    # graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
    # nx.set_node_attributes(G=graph, name="ndd", values={0: True, 1: False, 2: False})
    # #
    # m = solve(graph, 2, 5, True)
    # for v in m.getVars():
    #     print(v)

    #
    # graph = nx.DiGraph()
    # graph.add_edges_from([(0, 1), (0, 2), (2, 3), (3, 2)])
    # nx.set_node_attributes(G=graph, name="ndd", values={0: True, 1: False, 2: False, 3:False})
    # m = solve(graph,
    #           max_chain_length=5,
    #           max_cycle_length=2,
    #           name_gbvars=True)
    #
    # for v in m.getVars():
    #     print(v)

    # graph = nx.DiGraph()
    # graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)])
    # nx.set_node_attributes(G=graph, name="ndd", values={0: True, 1: False, 2: False, 3: False})
    # m = solve(graph,
    #           max_chain_length=5,
    #           max_cycle_length=2,
    #           name_gbvars=True)
    #
    # for v in m.getVars():
    #     print(v)
    #
    # graph = nx.DiGraph()
    # graph.add_edges_from([(0, 3), (1, 3), (2, 3)])
    # nx.set_node_attributes(G=graph, name="ndd", values={0: True, 1: True, 2: True, 3: False})
    # m = solve(graph,
    #           max_chain_length=1,
    #           max_cycle_length=2,
    #           name_gbvars=True)
    #
    # for v in m.getVars():
    #     print(v)
