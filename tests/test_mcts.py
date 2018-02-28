import pytest
from copy import deepcopy
from matching.utils.env_utils import snapshot, get_actions
from matching.tree_search import mcts_with_opt_rollout as mcts
from matching.environment.saidman_environment import SaidmanKidneyExchange

@pytest.fixture()
def root():
    env = SaidmanKidneyExchange(entry_rate=5, death_rate=.1, time_length=10)
    return mcts.Node(parent = None,
                t = 0,
                reward = 1,
                env = snapshot(env, 0),
                taken = None,
                actions = get_actions(env, 0),
                priors = None)
    

    
    
def test_rollout(root):
    """ ? """
    rs = mcts.rollout(root.env,
                 t_begin = 0,
                 t_end = 10,
                 taken = (0,1))
    
    
    
    
    


def test_expand(root):
    assert len(root.children) == 0
    parent = root
    while root.expandable:
        c = mcts.expand(parent)
        if c.taken is None: # "Advance"
            assert c.t == parent.t+1
            assert None in c.expandable
            assert c.expandable != parent.expandable
        else: # "Stay"
            assert c.t == parent.t
            for v in c.taken:
                for exp in c.expandable:
                    assert exp is None or v not in exp
            assert set(c.actions).issubset(parent.actions)


    

def test_backup(root):
    # Backing up from root
    expected_visits = 1
    expected_rewards = 1
    for i in range(5):
        assert root.visits == expected_visits
        assert root.reward == expected_rewards
        mcts.backup(root, 10)
        expected_visits += 1
        expected_rewards += 10
        
    # Backing up from child
    child = mcts.Node(parent= root,
             t = 1,
             reward = 1,
             env = root.env,
             taken = None,
             actions = root.actions)
    assert child.reward == 1
    assert child.visits == 1
    root.children.append(child)
    mcts.backup(child, 12)
    assert child.reward == 13
    assert child.visits == 2
    assert root.visits == expected_visits + 1
    assert root.reward == expected_rewards + 12
    
    
    
    
def test_stay(root):
    acts = set([(0,1), (1,2), (1,3), (5,6), None])
    root.expandable = acts
    root.actions = acts
    root.env.removed_container[0] = {10, 11}
    child = mcts.stay(root, (0,1))
    assert child.t == 0
    assert child.reward == 1
    assert child.env.removed_container[0] == {0, 1, 10, 11}
    assert child.expandable == {None, (5,6)}
    assert set(child.actions).issubset(root.actions)
    assert id(child.env) != id(root.env)

    
    
    
def test_advance(root):
    acts = set([(0,1), (1,2), (1,3), (5,6), None])
    root.expandable = acts
    root.actions = acts
    root.env.removed_container[0] = {10, 11}
    child = mcts.advance(root)
    assert child.t == 1
    assert child.reward == 1
    assert child.env.removed_container[0] == {10, 11}
    assert child.env.removed_container[1] == set()
    assert child.expandable != acts
    assert child.actions != acts
    assert set(child.actions) == set(child.expandable)
    assert id(child.env) != id(root.env)










    

