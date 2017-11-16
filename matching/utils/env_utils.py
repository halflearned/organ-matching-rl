from random import shuffle
from copy import deepcopy
#from matching.environment.optn_environment import OPTNKidneyExchange

def get_actions(env, t):
    cycles = two_cycles(env, t) 
    actions = list(map(tuple, cycles))
    actions.append(None)
    shuffle(actions)
    return actions


def two_cycles(env, t):
    nodes = list(env.get_living(t))
    cycles = []
    for i, u in enumerate(nodes):
        for w in nodes[i:]:
            if env.has_edge(u,w) and env.has_edge(w,u):
                cycles.append((u,w))
    return cycles



    
def snapshot(env, t):

    new_env = env.__class__(entry_rate = env.entry_rate,
                            death_rate = env.death_rate,
                            time_length = env.time_length,
                            seed = env.seed,
                            populate = False)


    subg = env.subgraph(env.get_living(t))
    new_env.add_nodes_from(subg.nodes(data = True))
    new_env.add_edges_from(subg.edges(data = True))
    
    new_env.removed_container = deepcopy(env.removed_container)
        
    return new_env

    
    
        