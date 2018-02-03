from random import shuffle
from re import findall
import numpy as np
from collections import defaultdict
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
    

def cumavg(x):
    time = np.arange(1, len(x) + 1)
    return np.cumsum(x)/time


def remove_taken(actions, taken):
    return [e for e in actions
            if e is None or 
                len(set(e).intersection(taken)) == 0]



def get_atrisk(env, t_begin = None, t_end = None):
    return [n for n,d in env.nodes(data = True) 
            if d["death"] >= t_begin and d["death"] < t_end]
  
    
def get_loss(env, t_begin, t_end, matched):
    n_living = len(env.get_living(t_begin, t_end))
    atrisk = get_atrisk(env, t_begin, t_end)
    nonsaved = set(atrisk).difference(matched)
    return len(nonsaved) / n_living
    


    
def snapshot(env, t):
    
    new_env = env.__class__(entry_rate = env.entry_rate,
                            death_rate = env.death_rate,
                            time_length = env.time_length,
                            seed = env.seed,
                            populate = False)


    nodelist = env.get_living(t)
    subg = env.subgraph(nodelist)
    new_env.add_nodes_from(subg.nodes(data = True))
    new_env.add_edges_from(subg.edges(data = True))
    
    rem_in_new_env = env.removed(t).intersection(new_env.nodes())
    new_env.removed_container = defaultdict(set) 
    new_env.removed_container[t] = rem_in_new_env
    
    # Forget death times
    try:
        new_env.data = env.data.loc[nodelist].copy()
        n = len(new_env.data)
        if n > 0:
            new_env.data["death"] =  t + np.random.geometric(new_env.death_rate, size = n)
    
        assert np.all(new_env.data.index == new_env.nodes)
    
    except AttributeError:
        for node in new_env.nodes:
            new_env.node[node]["death"] = t + \
                                np.random.geometric(new_env.death_rate) - 1
        
    
    return new_env

    
    
        