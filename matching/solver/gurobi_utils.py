import gurobipy as gb
from itertools import permutations, combinations
from collections import defaultdict
from copy import deepcopy
import numpy as np
import re
from tqdm import tqdm


def get_living_subset(data, begin, end = None):
    if end is None: end = begin
    living = (data["entry"] <= begin) & (data["death"] >= end)
    return set(data[living].index)


def get_critical_subset(data, t):
    critical = data["death"] == t
    return set(data[critical].index)


def get_noncyclical_permutations(idx, k):
    """ 
     Returns all k-permutations of idx that are 
     unique up to cyclical rotation.
    """
    for cmb in combinations(idx, k):
        head = cmb[:1]
        tail = cmb[1:]
        for p in permutations(tail):
            yield head + p
        
    
def can_give(blood1, blood2):
    """ Checks for bloodtype compatibility """
    return (blood1 == 0) | (blood2 == 3) | (blood1 == blood2)




def get_available_cycles(data, t, cycle_length):
    """ Returns available cycles of desired max_cycle_length on demand """
    
    living_idx = get_living_subset(data, t)
    all_cycles = list(get_noncyclical_permutations(living_idx, cycle_length))       
    if len(all_cycles) == 0:
        return []
    
    all_cycles = np.array(list(all_cycles))
    
    valid = []
    for i in range(cycle_length):
        j = (i + 1) % cycle_length
        d = data.loc[all_cycles[:,i], "donor"]
        p = data.loc[all_cycles[:,j], "patient"]
        valid.append(can_give(d.values, p.values))
        
    valid = np.vstack(valid).all(0)
    
    return all_cycles[valid]



            

def solve_akbarpour(data,
                    max_cycle_length,
                    max_cycles_by_period,
                    mode = "greedy"):
    
    """ 
    Solves Greedy and Patients algorithms 
    Source: Akbarpour, Li, Oveis-Gharan (2017)
    """
    
    data = deepcopy(data)
    t_max = int(max(data["entry"]) + 1)
    chosen = {}
    
    for t in tqdm(range(int(t_max)), desc = "Solving"):
        
        # Initialize model
        m = gb.Model()
        m.setParam("OutputFlag", 0)
        m.setParam("Threads", 1)
        cycle_constraints = defaultdict(list)
        max_cycles_by_period_constr = []
        
        # Generate cycle constraints
        variables = []
        if mode == "patient":
            critical = get_critical_subset(data, t)
            
        for b in range(2, max_cycle_length + 1):
            cycles = get_available_cycles(data, t, b)
            names = ["Time: {} | Cycle: {}".format(t, "-".join((str(i) for i in cyc))) for cyc in cycles]
            for name, cyc in zip(names, cycles):
                if mode == "patient":
                    if len(cyc.intersection(critical)) == 0:
                        continue
                x_cycle = m.addVar(vtype=gb.GRB.BINARY, name=name)
                variables.append(x_cycle)
                for v in cyc:
                    cycle_constraints[v].append(x_cycle)
                if max_cycles_by_period:
                    max_cycles_by_period_constr.append(x_cycle)
                
        
                
        
        # Add constraints to model
        for v in cycle_constraints:
            m.addConstr(gb.quicksum(cycle_constraints[v]) <= 1, 
                        name = "v_%d" % v) 
            
        if max_cycles_by_period:
            m.addConstr(gb.quicksum(max_cycles_by_period_constr) <= max_cycles_by_period,
                        name = "max_cycles_by_period")
            
        m.update()
        # Objective
        m.setObjective(gb.quicksum(variables), gb.GRB.MAXIMIZE)
        m.optimize()
        
        # Drop these variables from data
        period_chosen = get_matched(m)
        data.drop(period_chosen, inplace = True)
        chosen[t] = set(period_chosen)
        
        
    return chosen


#@profile
def solve_optimal(data, 
                  max_cycle_length, 
                  max_cycles_by_period = None,
                  **param_kwargs):

    data = data.astype(int)
    
    t_max = int(max(data["entry"]) + 1)
    
    # Initialize model
    m = gb.Model()
    m.setParam("OutputFlag", 0)
    m.setParam("Threads", 1)
    for k in param_kwargs:
        m.setParam(k, param_kwargs[k])
    
    # Create variables and reserve constraints
    variables = []
    cycle_constraints = defaultdict(list)
    max_cycles_by_period_constr  = defaultdict(list)
    
    for t in tqdm(range(int(t_max)),  desc = "Generating cycles"):
        for b in range(2, max_cycle_length + 1):
            cycles = get_available_cycles(data, t, b)
            names = ["Time: {} | Cycle: {}".format(t, "-".join((str(i) for i in cyc))) for cyc in cycles]
            for name, cyc in zip(names, cycles):
                x_cycle = m.addVar(vtype=gb.GRB.BINARY, 
                                   name=name)
                variables.append(x_cycle)
                for v in cyc:
                    cycle_constraints[v].append(x_cycle)
                if max_cycles_by_period:
                    max_cycles_by_period_constr[t].append(x_cycle)
        
    m.update()       
    
    print("Preparing constraints and objective.")
    for t in range(t_max):
        if max_cycles_by_period:
            m.addConstr(gb.quicksum(max_cycles_by_period_constr[t]) <= max_cycles_by_period,
                    name = "max_cycles_by_period_%d" % t)
            
    # Add constraints to model
    for v in cycle_constraints:
        m.addConstr(gb.quicksum(cycle_constraints[v]) <= 1, 
                    name = "v_%d" % v)
        
    # Objective
    m.setObjective(gb.quicksum(variables), gb.GRB.MAXIMIZE)
    
    
    print("Solving...", end = "")
    m.optimize()
    print("Done!")
    return m
    

def get_matched(model):
    chosen = []
    for c in model.getVars():
        if c.x != 0:
            cycle = c.VarName.split("|")[1]
            nodes = re.findall("[0-9]+", cycle)
            chosen.extend([int(n) for n in nodes])
    return list(set(chosen))



def get_chosen_cycles(model, reindex_map = None):
    cycles = defaultdict(set)
    for c in model.getVars():
        if c.x != 0:
            time, *nodes = map(int, re.findall("[0-9]+", c.varName))
            for v in nodes: 
                if reindex_map is not None: v = reindex_map[v]
                cycles[time].add(v)
    return cycles


    

#%%
#if __name__ == "__main__":
    
    
#    from environment import KidneyExchange
#    from random import choice
#    
#    dr     = .1 #choice([.01, .05, .1, .2, .3, .5])
#    er     = 5  #choice([1, 2, 3, 4, 5, 10, 20])
#    maxlen = 10*int(er/dr)
#    max_cycle_length = 2  #choice([2, 3, 4])
#    tc     = 5  #choice([1, 2, 3, 4, 5, 6])
#  
#    env = KidneyExchange(1, maxlen, death_rate = dr, entry_rate = er)
#    env.reset()
#    t_max = env.time_max_cycle_length
#    data = env.data[0].astype(int)
#    n = data.shape[0]
#    
#    opt = solve_optimal(data, max_cycle_length, tc)
#    opt_matched = get_matched(opt)
#    opt_alive_at_end = sum((data["death"] > t_max) & ~data.index.isin(opt_matched))
#    
#    p_matched = solve_akbarpour(data, max_cycle_length, "patient", tc)
#    p_alive_at_end = sum((data["death"] > t_max) & ~data.index.isin(p_matched))
#    
#    g_matched = solve_akbarpour(data, max_cycle_length, "greedy", tc)
#    g_alive_at_end = sum((data["death"] > t_max) & ~data.index.isin(g_matched))
#    
#    avg_entry = er * t_max
#    
#    opt_loss = 1 - (len(opt_matched) + opt_alive_at_end)/avg_entry
#    g_loss = 1 - (len(g_matched) + p_alive_at_end)/avg_entry
#    p_loss = 1 - (len(p_matched) + g_alive_at_end)/avg_entry
#    
#    print("OPT vs Greedy ({}, {}, {})".format(dr, er, max_cycle_length))
#    print("Loss (OPT): \t {:5.3f}".format(opt_loss))
#    print("Loss (GREEDY): {:5.3f}".format(g_loss))
#    print("Loss (PATIENT): {:5.3f}".format(p_loss))
#    
#    #with open("opt_vs_greedy.csv", "a") as f:
#    #    f.write("{},{},{},{},{:5.3f},{:5.3f},{:5.3f}\n"\
#    #    .format(dr, er, max_cycle_length, tc, len(opt)/n, len(g)/n, len(p)/n))
#  
#    
