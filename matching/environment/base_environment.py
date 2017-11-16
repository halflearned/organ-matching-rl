#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 15:08:29 2017

@author: vitorhadad
"""
import abc
import numpy as np
import networkx as nx
from collections import defaultdict
from matching.solver.trimble_solver.kidney_digraph import Digraph


def draw(p_dict, n = 1):
    return np.random.choice(range(len(p_dict)), 
                            p = list(p_dict.values()),
                            size = n)
    


    
class BaseKidneyExchange(nx.DiGraph, abc.ABC):
    

    def __init__(self, 
                 entry_rate,
                 death_rate,
                 time_length,
                 seed = None,
                 populate = True):
        
        
        nx.DiGraph.__init__(self)
        
        self.entry_rate = entry_rate
        self.death_rate = death_rate
        self.time_length = time_length
        self.seed = seed
        self.removed_container = defaultdict(set)
        
        if populate: self.populate()
        
        
    def removed(self, t):
        output = set()
        for k, vs in self.removed_container.items():
            if k <= t:
                output.update(vs)
        return set(output)
        
    
    
    def A(self, t, dtype = "numpy"):
        nodelist = self.get_living(t, indices_only = True)
        if dtype == "sparse":
            return nx.adjacency_matrix(self, nodelist)
        elif dtype == "numpy":
            return np.array(nx.to_numpy_matrix(self, nodelist))
        elif dtype == "pandas":
            return nx.to_pandas_dataframe(self, nodelist)
        else:
            raise ValueError("Unknown dtype")
        
    
        
    @abc.abstractmethod
    def X(self, t):
        pass
    
    @abc.abstractmethod
    def draw_node_features(self, t_begin, t_end):
        pass
    
    @abc.abstractmethod
    def draw_edges(self, source_nodes, target_nodes):
        pass
    


    def populate(self, t_begin = None, t_end = None, seed = None):
        
        if t_begin is None:
            t_begin = 0
        if t_end is None:
            t_end = self.time_length
        np.random.seed(seed)        
        
        self.erase_from(t_begin)
        max_cur_id = max(self.nodes(), default = 0)
        
        nodefts = self.draw_node_features(t_begin, t_end)
        new_ids = tuple(range(max_cur_id, max_cur_id + len(nodefts)))
        self.add_nodes_from(zip(new_ids, nodefts))
        #import pdb; pdb.set_trace()
        try:
            old_ids = list(self.nodes())
            
            oldnew_edges = self.draw_edges(old_ids, new_ids)
            self.add_edges_from(oldnew_edges, weight = 1)
            
            newold_edges = self.draw_edges(new_ids, old_ids)
            self.add_edges_from(newold_edges, weight = 1)
            
            newnew_edges = self.draw_edges(new_ids, new_ids)
            self.add_edges_from(newnew_edges, weight = 1)
        except:
            import pdb; pdb.set_trace()
        
        
    
    
    
    def validate_cycle(self, cycle):
        n = len(cycle)
        for i in range(n):
            j = (i + 1) % n
            if not self.can_give(cycle[i], cycle[j]):
                return False
        return True
    
        

    def can_give(self, i, j):
        return self.has_edge(i, j)
    
    
    
    def get_dying(self, t, indices_only = True):
        if indices_only:
            return [n for n,d in self.nodes(data = True) 
                    if d["death"] == t
                    and n not in self.removed(t)]
        else:
            return [(n,d) for n,d in self.nodes(data = True) 
                    if d["death"] == t
                    and n not in self.removed(t)]
            
            
        
    def erase_from(self, t):
        """
        Erases all with entry >= t
        """
        to_remove = [n for n,d in self.nodes(data = True) if d["entry"] >= t]
        self.remove_nodes_from(to_remove)
        for k in self.removed_container:
            if k > t:
                self.removed_container[k].clear()
            
            
    
    
    def get_living(self, t_begin, t_end = None, indices_only = True):
        if t_end is None: t_end = t_begin
        if indices_only:
            return [n for n,d in self.nodes(data = True) 
                    if d["entry"] <= t_end and d["death"] >= t_begin 
                    and n not in self.removed(t_begin)]
        else:
            return [(n,d) for n,d in self.nodes(data = True) 
                    if d["entry"] <= t_end and d["death"] >= t_begin
                    and n not in self.removed(t_begin)]
    
    
    def reindex_to_absolute(self, vs, t):
        living = self.get_living(t, indices_only = True)
        reindexed = [living[i] for i in vs]
        map_back = dict(zip(reindexed, vs))
        return reindexed, map_back
        
        
    def reindex_to_period(self, vs, t):
        living = self.get_living(t, indices_only = True)
        reindexed = np.argwhere(np.isin(living, vs)).flatten()
        return reindexed
        
    
    
    
    # Use this method instead of nx.relabel_nodes
    def relabel_nodes(self, mapping):
        # ??? What if this init had other parameters ???
        relabeled_env = self.__class__(self.entry_rate,
                           self.death_rate,
                           self.time_length,
                           seed = None,
                           populate = False)
        
        for i, attrs in self.nodes(data=True):
            relabeled_env.add_node(mapping[i], **attrs)
            
        for i, nbrs in self.adjacency():
            for j, attrs in nbrs.items():
                relabeled_env.add_edge(mapping[i], mapping[j], **attrs)
        
        return relabeled_env
            
        
    
    
    def to_trimble_digraph(self, g, weight = "weight"):
        vtx_count = g.number_of_nodes()
        digraph = Digraph(vtx_count)
        for i, j, data in g.edges(data = True):
            digraph.add_edge(data["weight"], digraph.vs[i], digraph.vs[j])
        
        return digraph
    