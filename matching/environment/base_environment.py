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
from matching.utils.data_utils import clock_seed


def draw(p_dict, n = 1):
    return np.random.choice(range(len(p_dict)), 
                            p = list(p_dict.values()),
                            size = n)
    


    
class BaseKidneyExchange(nx.DiGraph, abc.ABC):
    

    def __init__(self,
                 entry_rate,
                 death_rate,
                 time_length,
                 seed=None,
                 populate=True,
                 fraction_ndd=0):
        
        
        nx.DiGraph.__init__(self)
        
        self.entry_rate = entry_rate
        self.death_rate = death_rate
        self.time_length = time_length
        self.fraction_ndd = fraction_ndd
        self.removed_container = defaultdict(set)
        self.seed = clock_seed() if seed is None else seed
            
        if populate: self.populate(seed = seed)
        
        
        
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
        if len(self.nodes()) > 0:
            next_id = max(self.nodes()) + 1
        else:
            next_id = 0
        
        old_ids = list(self.nodes())
        
        nodefts = self.draw_node_features(t_begin, t_end)
        new_ids = tuple(range(next_id, next_id+len(nodefts)))
        
        self.add_nodes_from(zip(new_ids, nodefts))
        
        newnew_edges = self.draw_edges(new_ids, new_ids)
        
        self.add_edges_from(newnew_edges, weight = 1)
        
        if len(old_ids):
            oldnew_edges = self.draw_edges(old_ids, new_ids)
            self.add_edges_from(oldnew_edges, weight = 1)
        
            newold_edges = self.draw_edges(new_ids, old_ids)
            self.add_edges_from(newold_edges, weight = 1)
        
        
    
        
        
    def attr(self, *attrs, nodes = None):
        if nodes is None:
            nodes = self.nodes()
            
        np_attrs = []
        for at in attrs:
            np_attrs.append(np.array([self.node[n][at] for n in nodes]).reshape(-1, 1))
        return np.hstack(np_attrs)
        
    
    
    def validate_cycle(self, cycle):
        n = len(cycle)
        for i in range(n):
            j = (i + 1) % n
            if not self.can_give(cycle[i], cycle[j]):
                return False
        return True
    
        

    def can_give(self, i, j):
        return self.has_edge(i, j)
    
    
        
            
        
    def erase_from(self, t):
        """
        Erases all with entry >= t
        """
        to_remove = [n for n,d in self.nodes(data = True) if d["entry"] >= t]
        
        # Necessary to make OPTNEnvironment work, but really bad code :(
        try:
            self.data = self.data.drop(to_remove, axis = 0)
        except AttributeError: # Ouch, that's bad code
            pass
        
        
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
        
    
    
        