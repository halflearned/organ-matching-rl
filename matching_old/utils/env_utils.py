from random import shuffle


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