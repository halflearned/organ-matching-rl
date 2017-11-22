from random import shuffle
from re import findall
from collections import defaultdict
import networkx as nx
#from matching.environment.optn_environment import OPTNKidneyExchange

def get_actions(env, t):
    cycles = two_cycles(env, t) 
    actions = list(map(tuple, cycles))
    actions.append(None)
    shuffle(actions)
    return actions


def two_cycles(env, t, nodes = None):
    if nodes is None and t is not None:
        nodes = list(env.get_living(t))
    cycles = []
    for i, u in enumerate(nodes):
        for w in nodes[i:]:
            if env.has_edge(u,w) and env.has_edge(w,u):
                cycles.append((u,w))
    return cycles



def two_cycles_from_nodes(env, nodes = None):
    cycles = []
    for i, u in enumerate(nodes):
        for w in env.neighbors(u):
            if env.has_edge(u,w) and env.has_edge(w,u):
                cycles.append((u,w))
    return cycles



def get_environment_name(env):
    return findall("[A-Za-z]+KidneyExchange", str(env.__class__))[0]
    


def remove_taken(actions, taken):
    return [e for e in actions
            if e is None or 
                len(set(e).intersection(taken)) == 0]



    
def snapshot(env, t):

    new_env = env.__class__(entry_rate = env.entry_rate,
                            death_rate = env.death_rate,
                            time_length = env.time_length,
                            seed = env.seed,
                            populate = False)


    subg = env.subgraph(env.get_living(t))
    new_env.add_nodes_from(subg.nodes(data = True))
    new_env.add_edges_from(subg.edges(data = True))
    
    rem_in_new_env = env.removed(t).intersection(new_env.nodes())
    new_env.removed_container = defaultdict(set) 
    new_env.removed_container[t] = rem_in_new_env
    
    return new_env

    
    
        