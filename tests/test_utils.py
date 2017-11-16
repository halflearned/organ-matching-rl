import pytest
import networkx as nx
import numpy as np
from itertools import product
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.utils.env_utils import two_cycles, remove_taken, snapshot

@pytest.fixture
def env():
    e = SaidmanKidneyExchange(
        entry_rate = 10,
        death_rate = .1,
        time_length = 50,
        populate = True)
    return e
    

def test_two_cycles_all(env):
    cycles = two_cycles(env, 4)
    for i, j in cycles:
        assert env.has_edge(i, j)
        assert env.has_edge(j, i)
        assert env.node[i]["entry"] <= env.node[j]["death"]
        assert env.node[j]["entry"] <= env.node[i]["death"]
        assert env.node[i]["d_blood"] == 0 or \
               env.node[j]["p_blood"] == 3 or \
               env.node[i]["d_blood"] == env.node[j]["p_blood"]
                   
    
    
    
@pytest.mark.parametrize("exp,tak,tru", [
    ({(0,1),(1,2),(3,4),None},   (3,4),  {(0,1), (1,2), None}),
    ({(0,1),(1,2),(3,4),None},   (1,2),  {(3,4), None}),
    ({(0,1),(0,2),(0,4),None},   (0,1),  {None}),
    ({(0,1),(1,2),(3,4),None},   (0,1),  {(3,4),None}),
])
def test_remove_taken(exp,tak,tru):
    assert set(remove_taken(exp, tak)) == tru
    
    
    
    
def test_snapshot(env):
    for t in range(env.time_length):
        snap = snapshot(env, t)
        assert env.removed_container == snap.removed_container
        assert env.removed_container is not snap.removed_container
        for n,d in env.nodes.data():
            if n in snap.nodes():
                assert snap.node[n] == env.node[n]
                assert snap.node[n] is not env.node[n]
                assert d["entry"] <= t
                assert d["death"] >= t
            else:
                assert d["death"] < t or d["entry"] > t
        
        
        
        
        
        
        
        
        
        
    
    