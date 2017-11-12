import pytest
from copy import deepcopy
from matching.tree_search.mcts import Node, MCTS
from matching.environment.saidman_environment import SaidmanKidneyExchange

@pytest.fixture()
def mcts():
    env = SaidmanKidneyExchange(entry_rate = 5,
                                death_rate = .1,
                                time_length = 10)
    return MCTS(env, 0)
    
    
    
def test_node_creation():
    env = SaidmanKidneyExchange(entry_rate = 5,
                                death_rate = .1,
                                time_length = 10)
    env0 = MCTS(env, 0).root.env
    env1 = MCTS(env, 3).root.env
    env2 = MCTS(env, 7).root.env
    last0 = env0.node[max(env0.nodes())]
    last1 = env1.node[max(env1.nodes())]
    last2 = env2.node[max(env2.nodes())]
    assert last0["entry"] == 0
    assert last1["entry"] == 3
    assert last2["entry"] == 7

    
    
def test_rollout(mcts):
    env = mcts.root.env
    env_copy = deepcopy(env)
    last_copy = env_copy.node[max(env_copy.nodes())]
    assert last_copy["entry"] == 0
    rs = mcts.rollout(mcts.root)
    last = env.node[max(env.nodes())]
    assert last["entry"] > 0
    assert last_copy["entry"] == 0
    
    
    
@pytest.mark.parametrize("exp,tak,tru", [
    ({(0,1),(1,2),(3,4),None},   (3,4),  {(0,1), (1,2), None}),
    ({(0,1),(1,2),(3,4),None},   (1,2),  {(3,4), None}),
    ({(0,1),(0,2),(0,4),None},   (0,1),  {None}),
    ({(0,1),(1,2),(3,4),None},   (0,1),  {(3,4),None}),
])
def test_remove_taken(mcts,exp,tak,tru):
    assert mcts.remove_taken(exp, tak) == tru
    
    

def test_expand(mcts):
    assert len(mcts.root.children) == 0
    parent = mcts.root
    while mcts.root.expandable:
        c = mcts.expand(parent)
        if c.taken is None: # "Advance"
            assert c.t == parent.t+1
            assert c.matched == set()
            assert None in c.expandable
            assert c.expandable != parent.expandable
        else: # "Stay"
            assert c.t == parent.t
            assert c.matched == set(c.taken)
            for v in c.taken:
                for exp in c.expandable:
                    assert exp is None or v not in exp
            assert c.actions == parent.actions


    

def test_backup(mcts):
    # Backing up from root
    expected_visits = 1
    expected_rewards = 0
    for i in range(5):
        assert mcts.root.visits == expected_visits
        assert mcts.root.reward == expected_rewards
        mcts.backup(mcts.root, 10)
        expected_visits += 1
        expected_rewards += 10
        
    # Backing up from child
    child = Node(parent= mcts.root,
             t = 1,
             reward = 0,
             env = mcts.root.env)
    assert child.reward == 0
    assert child.visits == 1
    mcts.root.children.append(child)
    mcts.backup(child, 12)
    assert child.reward == 12
    assert child.visits == 2
    assert mcts.root.visits == expected_visits + 1
    assert mcts.root.reward == expected_rewards + 12
    
    
    
    
def test_stay(mcts):
    acts = set([(0,1), (1,2), (1,3), (5,6), None])
    mcts.root.expandable = acts
    mcts.root.actions = acts
    mcts.root.matched = {10, 11}
    child = mcts.stay(mcts.root, (0,1))
    assert child.t == 0
    assert child.reward == 2
    assert child.matched == set((10, 11, 0, 1))
    assert child.mcl == 2
    assert child.expandable == {None, (5,6)}
    assert child.actions == acts
    assert id(child.env) != id(mcts.root.env)
    for i in range(mcts.root.env.number_of_nodes()):
        n_root = mcts.root.env.node[i]
        n_child = child.env.node[i]
        assert n_root == n_child
    
    
    
    
def test_advance(mcts):
    acts = set([(0,1), (1,2), (1,3), (5,6), None])
    mcts.root.expandable = acts
    mcts.root.actions = acts
    mcts.root.matched = {10, 11}
    child = mcts.advance(mcts.root)
    assert child.t == 1
    assert child.reward == 0
    assert child.matched == set((10, 11))
    assert child.mcl == 2
    assert child.expandable != acts
    assert child.actions != acts
    assert child.actions == child.expandable
    assert id(child.env) != id(mcts.root.env)
    for i in range(child.env.number_of_nodes()):
        n_root = mcts.root.env.node[i]
        n_child = child.env.node[i]
        if n_root["entry"] == 0:
            print("ROOT NODE")
            assert n_root == n_child
        else:
            print("TESTHING DIFF")
            assert n_root != n_child










    

