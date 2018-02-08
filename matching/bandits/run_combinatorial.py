
from sys import platform, argv
from random import choice
import numpy as np
from tqdm import trange

from matching.utils.env_utils import two_cycles
from matching.solver.kidney_solver2 import optimal, greedy
from matching.utils.data_utils import  clock_seed, get_n_matched

from matching.environment.abo_environment import ABOKidneyExchange
from matching.bandits.combinatorial import CombinatorialBandit

env = ABOKidneyExchange(4, .1, 101)

rewards = np.zeros(env.time_length)

opt = optimal(env)
gre = greedy(env)
o = get_n_matched(opt["matched"], 0, env.time_length)
g = get_n_matched(gre["matched"], 0, env.time_length)

for t in trange(env.time_length):

    cycles = two_cycles(env, t)
    print("len(cycles): ", len(cycles))
    if len(cycles) > 0:
        algo = CombinatorialBandit(env, t, iters_per_arm=10, max_match = 4)
        algo.simulate()
        best = algo.choose()
        env.removed_container[t].update(best)
        rewards[t] = len(best)

    if t == env.time_length - 1:
        rewards[t] += len(optimal(env, t, t)["matched_pairs"])

    print(t, np.sum(rewards[:t + 1]),
          np.sum(g[:t + 1]),
          np.sum(o[:t + 1]))


