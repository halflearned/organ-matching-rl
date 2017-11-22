#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:54:49 2017

@author: vitorhadad
"""

import gurobipy as gb
from tqdm import tqdm
from collections import defaultdict
import re
from copy import deepcopy
from itertools import chain


__all__ = ["KidneySolver"]

# Yes, I recognize this function could be a lot shorter.
def _merge_outputs(dicts):
    merged = {}
    merged["matched"] = defaultdict(list)
    merged["obj_values"] = defaultdict(int)
    merged["variables"] = defaultdict(list)
    for d in dicts:
        for k in d["matched"]:
            merged["matched"][k] += d["matched"][k]
        for k in d["obj_values"]:
            merged["obj_values"][k] += d["obj_values"][k]
        for k in d["variables"]:
            merged["variables"][k] += d["variables"][k]
    return merged


class KidneySolver:
    
    def __init__(self,
                 max_cycle_length,
                 max_cycles_per_period,
                 burn_in = 0):
        
        self.max_cycle_length = max_cycle_length
        self.max_cycles_per_period = max_cycles_per_period
        self.time_length = env.time_length
        self.burn_in = burn_in

        
   
    #TODO: Replace get_*_cycles by general-purpose algorithm
    def get_two_cycles(self, env, nodes = None):
        
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
    
    
    def get_three_cycles(self, env, nodes = None):
        if nodes is not None:
            subgraph = env.subgraph(nodes)
        else:
            subgraph = env
            
        edges = subgraph.edges()
        nodes = subgraph.nodes()
        wgt = nx.get_edge_attributes(subgraph, 'weight')
        output = []
        values = []
        for u, v in edges:
            for w in nodes:
                if w >= u or w >= v:
                    break
                if subgraph.has_edge(v, w) and subgraph.has_edge(w, u):
                    output.append((u, w, v))
                    values.append(wgt[(u, v)] + wgt[(v, w)] + wgt[(w, u)])
        return output, values
    
    
    def get_cycles(self, env, nodes):
        if self.max_cycle_length != 2 and self.max_cycle_length != 3:
            raise ValueError("Not supported cycle length")
            
        cycles = []
        weights = []

        c2s, w2s = self.get_two_cycles(env, nodes)
        cycles.extend(c2s)
        weights.extend(w2s)
        if self.max_cycle_length == 3:
            c3s, w3s = self.get_three_cycles(env, nodes)
            cycles.extend(c3s)
            weights.extend(w3s)
            
            
        return cycles, weights        
    
    
    
    def solve_greedy(self, env):
    
        """ 
        Solves Greedy and Patients algorithms 
        Source: Akbarpour, Li, Oveis-Gharan (2017)
        """
        
        env_copy = deepcopy(env)
        outputs = []
        
        for t in tqdm(range(self.time_length), 
                      desc = "Generating cycles"):
            
            # One model per period
            m = gb.Model()
            m.setParam("OutputFlag", 0)
            m.setParam("Threads", 1)
            cycle_constraints = defaultdict(list)
            max_cycles_per_period_constr = []
            
            # Choose only nodes living at t
            nodes = env_copy.get_living(t)
            
            # Generate cycle constraintsis
            
            cycles, weights = self.get_cycles(env_copy, nodes)
            
            assert len(cycles) == len(weights)
            
            variables = []
            
            for w, cyc in zip(weights, cycles):
                
                name = "Time: {} | Weight: {} | Cycle: {}".format(t, w, str(cyc))
            
                x_cycle = m.addVar(vtype=gb.GRB.BINARY, name=name)
                
                variables.append(x_cycle)
                
                for v in cyc:
                    cycle_constraints[v].append(x_cycle)
                
                if self.max_cycles_per_period:
                    max_cycles_per_period_constr.append(x_cycle)
                    

            # Add constraints to model
            for v in cycle_constraints:
                m.addConstr(gb.quicksum(cycle_constraints[v]) <= 1, 
                            name = "v_%d" % v) 
                
            if self.max_cycles_per_period is not None:
                m.addConstr(gb.quicksum(max_cycles_per_period_constr) <= self.max_cycles_per_period,
                            name = "max_cycles_per_period")
                      
                    
            m.update()
            
            # Objective
            m.setObjective(gb.quicksum([w*v for w,v in zip(weights, variables)]),
                           gb.GRB.MAXIMIZE)
            m.optimize()
            

            
            # Drop these variables from data
            period_matched = self.get_matched_nodes(m)
            env_copy.remove_nodes_from(period_matched[t])
            
            outputs += [self._get_output(m)]
            
            
        #import pdb; pdb.set_trace()
           
        out = _merge_outputs(outputs)
            
        out["obj_value"] = sum([out["obj_values"][t] for t in out["obj_values"] \
                                   if t >= self.burn_in])
        return out
    
    
            

    def solve_optimal(self, env):
        
        # Initialize model
        m = gb.Model()
        m.setParam("OutputFlag", 0)
        m.setParam("Threads", 1) 
        
        # Create variables and reserve constraints
        cycle_constraints = defaultdict(list)
        max_cycles_per_period_constr = defaultdict(list)
        variables = []
        weights = []
        
        for t in tqdm(range(self.time_length), 
                      desc = "Generating cycles"):
            
            nodes = env.get_living(t)
            
            cycles, ws = self.get_cycles(env, nodes)
            weights.extend(ws)
            
            
            for w, cyc in zip(weights, cycles):
                
                name = "Time: {} | Weight: {} | Cycle: {}".format(t, w, str(cyc))
                
                x_cycle = m.addVar(vtype=gb.GRB.BINARY, name=name)
                
                variables.append(x_cycle)
                
                for v in cyc:
                    cycle_constraints[v].append(x_cycle)
                
                if self.max_cycles_per_period:
                    max_cycles_per_period_constr[t].append(x_cycle)
        
        
        print("Preparing constraints and objective.")
         # Add constraints to model
        for v in cycle_constraints:
            m.addConstr(gb.quicksum(cycle_constraints[v]) <= 1, name = "v_%d" % v)
            
        if self.max_cycles_per_period:    
            for t in range(self.time_length):
                m.addConstr(gb.quicksum(max_cycles_per_period_constr[t]) <= self.max_cycles_per_period,
                            name = "max_cycles_per_period_%d" % t)
    
    
        
        # Objective
        m.update()
        
        m.setObjective(gb.quicksum([w*v for w, v in zip(weights, variables)]),
                       gb.GRB.MAXIMIZE)
        
        print("Solving...", end = "")
        m.optimize()
        print("Done!")
        
        out = self._get_output(m)
        
        out["obj_value"] = sum([out["obj_values"][t] for t in out["obj_values"] \
                                   if t >= self.burn_in])
                
        return out
    

        
    
    def _get_output(self, m):
        obj_values = defaultdict(int)
        matched = defaultdict(list)
        variables = defaultdict(list)
        for k, cyc in enumerate(m.getVars()):
            t, w, *vs = [int(r) for r in re.findall("[0-9]+", cyc.varName)]
            if cyc.x > 0:
                obj_values[t] += w
                matched[t] += vs
                variables[t] += [cyc.varName]
            else:
                variables[-1] += [cyc.varName]
        
        output = {"obj_values": obj_values,
                  "matched": matched,
                  "variables": variables}
        return output
            
            
    
    
    
    def get_post_burnin_obj_value(self, weights, m):
        obj_value = {}
        for k, cyc in enumerate(m.getVars()):
            if cyc.x > 0:
                t, *vs = [int(r) for r in re.findall("[0-9]+", cyc.varName)]
                if t >= self.burn_in:
                    obj_value[t] = weights[k]
        return obj_value
    
        
    def get_matched_nodes(self, m):
        nodes = defaultdict(list)
        for cyc in m.getVars():
            if cyc.x > 0:
                t, *vs = [int(r) for r in re.findall("[0-9]+", cyc.varName)]
                nodes[t].extend(vs)
        return nodes
            
        
        
        
#%%
if __name__ == "__main__":
    
    #TODO: Move tests to pytest later
    from saidman_environment import SaidmanKidneyExchange
    import numpy as np
    import networkx as nx
    from itertools import chain
    
    for i in range(100):
        env = SaidmanKidneyExchange(entry_rate   = 5,
                                    death_rate  = 0.1,
                                    time_length = 100,
                                    seed = i)
    
        solver = KidneySolver(max_cycle_length = 2,
                              max_cycles_per_period = None,
                              burn_in = 0)  
        
        opt = solver.solve_optimal(env)
        grd = solver.solve_greedy(env)
        
        # OPT is better
        assert opt["obj_value"] >= grd["obj_value"]
    
        print("GREEDY: ", grd["obj_value"])
        print("OPTIMAL: ", opt["obj_value"])
        
        if grd["obj_value"] != 0:
            print("Ratio", opt["obj_value"]/grd["obj_value"])
        

    
        for out in [opt, grd]:
            
            matched = list(chain(*out["matched"].values()))
            
            # Uniqueness
            assert len(matched) == len(set(matched))
            
            # Assuming weight = 1
            ws = 0
            for k in out["variables"]:
                if k >= solver.burn_in:
                    for v in out["variables"][k]:
                        _, w, *vs = [int(x) for x in re.findall("[0-9]+", v)]
                        # Each cycle weight equals # of matched in cycle
                        assert w == len(vs)
                        ws += w
                
            # Every cycle is valid
            for k in out["variables"]:
                if k >= solver.burn_in:
                    for v in out["variables"][k]:
                        t, _, *vs = [int(x) for x in re.findall("[0-9]+", v)]
                        # Each cycle weight equals # of matched in cycle
                        assert env.validate_cycle(vs)
                        living = env.get_living(t)
                        for n in vs: 
                            assert n in living
                        

            # Total weight is obj value
            assert ws == out["obj_value"]
            
            # Total obj value is sum of obj_value
            assert ws == sum(out["obj_values"].values())

            # Total matched equals obj_value
            assert len(matched) == out["obj_value"]
    
            
    
    
#%% Shuffler
#    
#    shuffled_env = deepcopy(env)
#    n = env.number_of_nodes()
#    true_index = range(n)
#    shuffled_index = np.random.permutation(n)
#    mapping_true_to_shuffled = dict(zip(true_index, shuffled_index))
#    mapping_shuffled_to_true = dict(zip(shuffled_index, true_index))
#    shuffled_env = env.relabel_nodes(mapping_true_to_shuffled)
#        
#    
    
    
    
    
    
    
    
    