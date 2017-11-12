import pytest
import numpy as np
from copy import deepcopy
from itertools import product
from matching.environment.saidman_environment import SaidmanKidneyExchange

@pytest.fixture
def env():
    return SaidmanKidneyExchange(
        entry_rate = 5,
        death_rate = .1,
        time_length = 50,
        seed = 12345,
        populate = True)


def test_erase_future(env):
    t = 5
    env_before = env
    env_after = deepcopy(env_before)
    env_after.erase_future(t)
    n = min(env_after.number_of_nodes(), 
            env_before.number_of_nodes())
    
    n_equals = 0
    for i in range(n):
        n1 = env_before.node[i]
        n2 = env_after.node[i]
        if n1["entry"] <= t:
            assert n1 == n2
        else:
            n_equals += n1 == n2
    assert n_equals < 5 # Some of them might be same by coincidence
    last_after_node = env_after.node[env_after.number_of_nodes()-1]
    assert last_after_node["entry"] == 5
    
        
def test_populate(env):
    for n,d in env.nodes(data = True):
        assert d["entry"] < d["death"]
        
        
        
        
def test_repopulate(env):
    t = 2
    T_new = 27
    env_before = deepcopy(env)
    env_after = deepcopy(env_before)
    env_after.populate(t_begin = t,
                       t_end = T_new)
    n = min(env_after.number_of_nodes(), 
            env_before.number_of_nodes())
    
    # No missing nodes, all consecutive
    assert set(env_after.nodes()) == set(range(env_after.number_of_nodes()))
    
    # Last node should enter at T-1
    last_after_node = env_after.node[env_after.number_of_nodes()-1]
    last_before_node = env_before.node[env_before.number_of_nodes()-1]
    assert last_before_node["entry"] == 49
    assert last_after_node["entry"] == T_new - 1

    # Most should be different after repopulating
    n_equals = 0
    for i in range(n):
        n1 = env_before.node[i]
        n2 = env_after.node[i]
        if n1["entry"] < t:
            assert n1 == n2
        else:
            n_equals += n1 == n2
    assert n_equals < 5 
    
    

def test_get_living(env):
    for t in range(50):
        liv = env.get_living(t)
        for i in liv:
            n = env.node[i]
            assert n["entry"] <= t
            assert n["death"] >= t
    
    
def test_is_blood_compatible():
    p_blood, d_blood = np.array(list(product([0,1,2,3], [0,1,2,3]))).T 
    b = SaidmanKidneyExchange.is_blood_compatible(d_blood, p_blood).astype(bool)
    for i, d in enumerate(d_blood):
        for j, p in enumerate(p_blood):
            if d == 0:
                assert b[i,j]
            elif p == 3:
                assert b[i,j]
            elif p == d:
                assert b[i,j]
            else:
                assert not b[i,j]
    
    
def test_is_contemporaneous():
    e, d = np.array(list(product([0,1,2,3], [1,2,3,4]))).T 
    c = SaidmanKidneyExchange.is_contemporaneous(e, d)
    for i in range(16):
        for j in range(16):
            if d[i] >= e[j] and e[i] <= d[j]:
                assert c[i,j]
            else:
                assert not c[i,j]

        