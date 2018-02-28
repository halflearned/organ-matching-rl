#
# n = 100
# graph = nx.gnp_random_graph(n, p=0.4, seed=1234, directed=True)  #env.subgraph(range(100))
# ndd = dict(zip(range(n), np.random.binomial(n=1, p=0.1, size=n)))
# nx.set_node_attributes(G=graph, name="ndd", values=ndd)
#
#
# m = gb.Model()
# #m.setParam("OutputFlag", 0)
# m.setParam("Threads", 1)
#
# in_edge_constraints = defaultdict(list)
# out_edge_constraints = defaultdict(list)
#
# edges = graph.edges()
# nodes = graph.nodes()
#
# edge_variables = dict(zip(edges, [m.addVar(vtype=gb.GRB.BINARY, name=str(e)) for e in edges]))
# flow_in_variables = dict(zip(nodes, [m.addVar(vtype=gb.GRB.BINARY, name=str(v)) for v in nodes]))
# flow_out_variables = dict(zip(nodes, [m.addVar(vtype=gb.GRB.BINARY, name=str(v)) for v in nodes]))
#
# for v, data in graph.nodes(data=True):
#
#     # Each vertex has at most one active in-edges
#     for e in graph.in_edges(v):
#         in_edge_constraints[v].append(edge_variables[e])
#
#     # Each vertex has at most one active out-edges
#     for e in graph.out_edges(v):
#         in_edge_constraints[v].append(edge_variables[e])
#
#     m.addConstr(gb.quicksum(in_edge_constraints[v]) == flow_in_variables[v])
#     m.addConstr(gb.quicksum(out_edge_constraints[v]) == flow_out_variables[v])
#
#     if data["ndd"]:
#         m.addConstr(flow_out_variables[v] <= 1)
#     else:
#         m.addConstr(flow_out_variables[v] <= flow_in_variables[v])
#         m.addConstr(flow_in_variables[v] <= 1)
#
# m.update()
# m.setObjective(gb.quicksum(list(edge_variables.values())), gb.GRB.MAXIMIZE)
# m.optimize()
#
# active_edges = [edge for edge, gb_var in zip(edges, m.getVars()) if gb_var.x > 0]
#
# assert len(active_edges) == (len(np.unique(active_edges)) / 2)
