import gurobipy as gb
from collections import defaultdict
from tqdm import trange


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
    return i_entry, i_death


def sojourn_overlap(graph, i, j):
    i_entry = graph.node[i]["entry"]
    i_death = graph.node[i]["death"] + 1
    j_entry = graph.node[j]["entry"]
    j_death = graph.node[j]["death"] + 1

    t_begin = max(i_entry, j_entry)
    t_end = min(i_death, j_death)
    return t_begin, t_end


def solve_with_time_constraints(graph,
                                max_cycle,
                                max_chain):
    m = gb.Model()
    m.setParam("Threads", 1)

    # Create all variables
    grb_vars = {}
    for edge in graph.edges():
        v, w = edge
        for t in range(*sojourn_overlap(graph, v, w)):
            if graph.node[v]["ndd"]:
                edge_var = m.addVar(vtype=gb.GRB.BINARY, name=str((v, w, 0, t)))
                grb_vars[(v, w, 0, t)] = edge_var
            else:
                for k in range(max_chain):
                    edge_var = m.addVar(vtype=gb.GRB.BINARY, name=str((v, w, k, t)))
                    grb_vars[(v, w, k, t)] = edge_var

    # Add constraints
    # TODO: Refactor this once it's working
    for q in graph.nodes():

        print(q, " of ", graph.number_of_nodes())

        # Capacity constraints
        if graph.node[q]["ndd"]:
            capacity = [x for (vp, wp, kp, tp), x in grb_vars.items() if vp == q]
            m.addConstr(gb.quicksum(capacity) <= 1)
        else:
            capacity = [x for (vp, wp, kp, tp), x in grb_vars.items() if wp == q]
            m.addConstr(gb.quicksum(capacity) <= 1)

        # A vertex that is not NDD...
        if not graph.node[q]["ndd"]:
            for s in range(*sojourn(graph, q)):
                # ...if at position 0, must have received in a previous time, any position
                cur = [x for (vp, wp, kp, tp), x in grb_vars.items() if vp == q and tp == s and kp == 0]
                prev = [x for (vp, wp, kp, tp), x in grb_vars.items() if wp == q and tp < s]
                if len(prev) or len(cur):
                    m.addConstr(gb.quicksum(cur) <= gb.quicksum(prev))

                # If not at position 0, must have received from position k-1 at this time period
                for k in range(max_chain):
                    cur = [x for (vp, wp, kp, tp), x in grb_vars.items() if vp == q and tp == s and kp == k]
                    prev = [x for (vp, wp, kp, tp), x in grb_vars.items() if wp == q and tp == s and kp == (k - 1)]
                    if len(prev) or len(cur):
                        m.addConstr(gb.quicksum(cur) <= gb.quicksum(prev))


    m.update()
    m.setObjective(gb.quicksum(list(grb_vars.values())), gb.GRB.MAXIMIZE)
    m.optimize()

    return m


if __name__ == "__main__":

    import networkx as nx

    from matching.trimble_solver.interface import greedy
    from matching.environment.abo_environment import ABOKidneyExchange

    for i in range(1):
        env = ABOKidneyExchange(5, 0.1, 10, seed=i, fraction_ndd=0.1)

        m = solve_with_time_constraints(env, 0, 2)
        xs = [eval(v.varName) for v in m.getVars() if v.x > 0]

        for k in sorted(xs, key=lambda x: x[3]):
            print(k, end="")
            if env.node[k[0]]["ndd"]:
                print("*")
            else:
                print("")

        received = {}
        donated = {}
        for v, w, k, t in xs:
            received[w] = t
            donated[v] = t
        for v in received:
            if v in donated:
                assert received[v] <= donated[v]

        gre = greedy(env, max_cycle=0, max_chain=2,
                     formulation="hpief_prime_full_red")
        assert gre["obj"] <= m.ObjVal


