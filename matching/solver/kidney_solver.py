#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:54:49 2017

@author: vitorhadad
"""

from collections import defaultdict
import networkx as nx
from copy import deepcopy
from itertools import chain
from matching.solver.trimble_solver.kidney_digraph import Digraph
from matching.solver.trimble_solver.kidney_ip import OptConfig
from matching.solver.trimble_solver.kidney_solver import solve_kep


__all__ = ["KidneySolver"]



class KidneySolver:
    
    def __init__(self,
                 max_cycle_length,
                 max_cycles_per_period = None,
                 burn_in = 0):
        
        self.max_cycle_length = max_cycle_length
        self.burn_in = burn_in
    
    
    
    def solve(self, env, t_begin, t_end = None, formulation = "cf"):
        t_end = t_end or t_begin
        nodelist = env.get_living(t_begin, t_end)
        subg = nx.subgraph(env, nodelist)
        labels_to_trimble = {n:i for i,n in enumerate(subg.nodes())}
        labels_from_trimble = {i:n for i,n in enumerate(subg.nodes())}
        subg = nx.relabel_nodes(subg, labels_to_trimble)
        trimble_digraph = self.to_trimble_digraph(subg)
        cfg = OptConfig(trimble_digraph, [], self.max_cycle_length, 0)
        sol = solve_kep(cfg, formulation)
        return self.parse_solution(sol, env, labels_from_trimble)
    
    
    

    def solve_subset(self, env, nodelist, formulation = "cf"):
        subg = nx.subgraph(env, nodelist)
        labels_to_trimble = {n:i for i,n in enumerate(subg.nodes())}
        labels_from_trimble = {i:n for i,n in enumerate(subg.nodes())}
        subg = nx.relabel_nodes(subg, labels_to_trimble)
        trimble_digraph = self.to_trimble_digraph(subg)
        cfg = OptConfig(trimble_digraph, [], self.max_cycle_length, 0)
        sol = solve_kep(cfg, formulation)
        return self.parse_solution(sol, env, labels_from_trimble)
    

    
    def optimal(self, env):
        s =  self.solve(env, 0, env.time_length)  
        s["obj"] = sum(len(c) for t,c in s["matched"].items() \
                     if t >= self.burn_in) #TODO: Weights
        return s
         
    
    
    def greedy(self, env, horizon = 0, t_begin = None, t_end = None):
        t_begin = 0 if t_begin is None else t_begin
        t_end = env.time_length if t_end is None else t_end
        
        env = deepcopy(env)
        output = {"matched": defaultdict(list),
                  "objs": defaultdict(lambda: 0),
                  "model": []}
        for t in range(t_begin, t_end):
            
            t_horizon = t + horizon
            s = self.solve(env, t, t_horizon)
            output["model"].append(s["model"])
            
            for t_match in range(t + 1):
                m = s["matched"][t_match]
                env.removed_container[t].update(m)
                output["objs"][t] += len(m) #TODO: Weights
                output["matched"][t].extend(m)

        output["obj"] = sum(output["objs"][t] for t in range(self.burn_in, env.time_length))
        
        return output
    
    
    def parse_solution(self, sol, env, labels):
        
        output = {}
        output["model"] = sol.ip_model
        output["matched"] = defaultdict(list)
        for c in sol.cycles:
            if labels:
                vs = [labels[v.id] for v in c]
            else:
                vs = [v.id for v in c]
            t_match = max([env.node[v]["entry"] for v in vs])
            output["matched"][t_match].extend(vs)
        return output
        

    
    def to_trimble_digraph(self, source_graph, weight = "weight"):
        vtx_count = source_graph.number_of_nodes()
        digraph = Digraph(vtx_count)
        for i, j, data in source_graph.edges(data = True):
            try:
                digraph.add_edge(data["weight"], digraph.vs[i], digraph.vs[j])
            except IndexError:
                import pdb; pdb.set_trace()
        
        return digraph
    

   

#%%
if __name__ == "__main__":
    
    import environment.saidman_environment.SaidmanKidneyExchange
    import numpy as np
    import networkx as nx
    from itertools import chain
    #%%

    env = SaidmanKidneyExchange(entry_rate  = 5,
                                death_rate  = 0.1,
                                time_length = 100,
                                seed = 1)

    solver = KidneySolver(max_cycle_length = 2,
                      burn_in = 0)  

    
    opt = solver.optimal(env)["obj"]
    greedy = solver.greedy(env, 0)["obj"]
    
        
        
        
        
        
        