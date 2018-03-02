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


def solve2(graph, max_cycle=2, max_chain=2):

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
                if k < max_chain - 1: 
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




if __name__ == "__main__":

    from matching.trimble_solver.kidney_digraph import read_digraph
    import matching.trimble_solver.kidney_ip as k_ip
    import matching.trimble_solver.kidney_ndds as k_ndds
    import matching.trimble_solver.kidney_utils as k_utils

    # Move these to tests later
    # graph = nx.DiGraph()
    # graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
    # nx.set_node_attributes(G=graph, name="ndd", values={0: True, 1: False, 2: False})
    # #
    # m = solve2(graph, 2, 5)
    # for v in m.getVars():
    #     print(v)
    #
    # graph = nx.DiGraph()
    # graph.add_edges_from([(0, 1), (0, 2), (2, 3), (3, 2)])
    # nx.set_node_attributes(G=graph, name="ndd", values={0: True, 1: False, 2: False, 3:False})
    # m = solve2(graph, max_chain=5)
    #
    # for v in m.getVars():
    #     print(v)

    # graph = nx.DiGraph()
    # graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)])
    # nx.set_node_attributes(G=graph, name="ndd", values={0: True, 1: False, 2: False, 3: False})
    # m = solve2(graph,
    #           max_chain=5,
    #           max_cycle=0)
    #
    # for v in m.getVars():
    #     print(v)

    graph = nx.DiGraph()
    graph.add_edges_from([(1, 3),
                          (1, 4),
                          (2, 4),
                          (3, 4),
                          (4, 5),
                          (5, 6),
                          (6, 5),
                          (6, 4)])

    nx.set_node_attributes(G=graph, name="ndd",
                           values={1: True, 2: True,
                                   3: False, 4: False,
                                   5: False, 6: False})
    m = solve2(graph,
              max_chain=5,
              max_cycle=0)

    for v in m.getVars():
        print(v)
