import networkx as nx

def get_two_cycles(env, nodes = None):
      
  if nodes is not None:
      subgraph = env.subgraph(nodes)
  else:
      subgraph = env
      
  edges = subgraph.edges(data = True)
  wgt = nx.get_edge_attributes(subgraph, 'weight')
  output = []
  weights = []
  for u, v, attr in edges:
      if u < v and subgraph.has_edge(v, u):
          output.append((u, v))
          weights.append(wgt[(u, v)] + wgt[(v, u)])
  return output, weights