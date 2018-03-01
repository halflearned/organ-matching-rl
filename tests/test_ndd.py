"""

Testing features associated with NDD donors

"""

import pytest
import numpy as np
from time import sleep

from matching.environment.abo_environment import ABOKidneyExchange
from matching.environment.saidman_environment import SaidmanKidneyExchange
from matching.environment.optn_environment import OPTNKidneyExchange


@pytest.fixture(params=[ABOKidneyExchange, SaidmanKidneyExchange])
def env(request):
    return request.param(entry_rate=5,
                       death_rate=.1,
                       time_length=50,
                       seed=12345,
                       populate=True,
                       fraction_ndd=0.1)


@pytest.fixture(params=[OPTNKidneyExchange])
def optn(request):
    return request.param(entry_rate=5,
                         death_rate=.1,
                         time_length=50,
                         seed=12345,
                         populate=True,
                         fraction_ndd=0.1)


def test_ndd_has_no_incoming_arrow(env):
    for v, w in env.edges():
        assert env.node[w]["ndd"] == 0


def test_ndd_X(env):
    for t in range(env.time_length):
        X = env.X(t, dtype="pandas")
        X_ndd = X.query("ndd > 0")
        if isinstance(env, OPTNKidneyExchange):
            assert np.all(X_ndd[['blood_A_pat', 'blood_AB_pat', 'blood_B_pat']] == 0)
        else:
            assert np.all(X_ndd[["pO", "pA", "pB"]] == 0)


def test_ndd_has_no_incoming_arrow_2(optn):
    for v, w in optn.edges():
        assert optn.data.loc[w, "ndd"] == 0


