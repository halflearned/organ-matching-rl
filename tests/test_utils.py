import pytest
import networkx as nx
import numpy as np
from itertools import product
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.utils.env_utils import get_two_cycles

@pytest.fixture
def env():
    e = SaidmanKidneyExchange(
        entry_rate = 10,
        death_rate = .1,
        time_length = 50,
        seed = 12345,
        populate = True)
    return e
    

def test_two_cycles_all(env):
    cycles, weights = get_two_cycles(env)
    for i, j in cycles:
        assert env.has_edge(i, j)
        assert env.has_edge(j, i)
        assert env.node[i]["entry"] <= env.node[j]["death"]
        assert env.node[j]["entry"] <= env.node[i]["death"]
        assert env.node[i]["d_blood"] == 0 or \
               env.node[j]["p_blood"] == 3 or \
               env.node[i]["d_blood"] == env.node[j]["p_blood"]
               
    # Number of cycles is correct
    A = np.array(nx.to_numpy_matrix(env))
    n_cycles = np.sum(np.diag(A @ A)) // 2
    assert len(cycles) == n_cycles
    
    