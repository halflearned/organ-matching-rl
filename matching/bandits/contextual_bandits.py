from collections import defaultdict
from collections import defaultdict
from typing import Union

# Scientific computing
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# Utilities
from matching.environment.base_environment import BaseKidneyExchange
from matching.utils.env_utils import snapshot


class Agent:

    def __init__(self,
                 arms: np.ndarray,
                 method: str,
                 context_history: np.ndarray,
                 estimator: BaseEstimator,
                 arm_history: np.ndarray,
                 reward_history: pd.Series,
                 keep_attributes: tuple = ()):

        self.arms = arms
        self.n_arms = len(self.arms)

        self.method = method
        self.keep_attributes = keep_attributes
        self.estimator = BaseEstimator

        self.context_history = context_history
        self.arm_history = arm_history
        self.reward_history = reward_history
        self.hyperparameter_history = defaultdict(list)

    def choose_arms(self, context: np.ndarray) -> np.ndarray:
        """
        Selects best arm for each context according to predicted statistics
        Parameters
        ----------
        context : np.ndarray
            New context for which arms will be chosen.
        Returns
        -------
        chosen : np.ndarray
            Best arm for each row in context.
        """
        if len(self.context_history) == 0:
            return np.random.randint(self.n_arms, size=len(context))

        stats = self.get_statistics(context)
        if self.method == "ucb":
            arm_idx = self.ucb(stats)
        elif self.method == "thompson":
            arm_idx = self.thompson(stats)
        else:
            ValueError("Unknown method")

        return arm_idx

    def ucb(self, stats: np.ndarray):
        """
        Best arm index according to UCB algorithm.
        Computes the confidence bands around each arm-context combination,
        then outputs the argmax arm for each context.
        Parameter
        ---------
        stats: pandas.DataFrame
            Contains [context, arm, mean, std] columns
        """
        n = len(self.context_history)  # TODO: Actually should be + len(new_context)?
        stats["ucb"] = stats["mean"] + np.sqrt(2 * np.log(n) * stats["std"])  # TODO: Check this fmla
        return stats.set_index("arm").groupby("context")["ucb"].idxmax()

    def thompson(self, stats: np.ndarray):
        """
        Thompson-sampled best arm index.
        Draws a reward from Normal(mu = stats["mean"], sigma = stats["std"])
        for each arm and context, then outputs the argmax arm for each context.
        Parameters
        ----------
        stats : np.ndarray
        Returns
        -------
        Thompson-sampled best arm index.
        """
        stats["draw"] = np.random.normal(stats["mean"], stats["std"])
        return stats.set_index("arm").groupby("context")["draw"].idxmax()

    def get_statistics(self, context: np.ndarray) -> np.ndarray:
        """
        This method must be implemented by a subclassing algorithm
        Returns
        -------
        A pandas DataFrame with columns [context, arm, mean, std] with the statistics
        for each context-arm combination. How these statistics are computed depends on the algorithm.
        """
        pass


class Agent:

    def __init__(self,
                 env: Union[BaseKidneyExchange],
                 t: int,
                 estimator=None,
                 cross_validation_method=None,
                 cross_vaidation_parameters=None):

        self.estimator = estimator
        self.cross_validation_method = cross_validation_method
        self.env = snapshot(env, t)
        self.t = t
        self.subgraph = self.env.subgraph(self.env.get_living(self.t))
        self.edges = self.subgraph.edges

    def get_matrix(self):
        vs, ws = np.array(list(self.subgraph.edges)).T
        X = self.env.X(self.t, dtype="pandas")
        return np.hstack([X.loc[vs].values,
                          X.loc[ws].values[:, :-2]])

    def choose_edge(self, context: np.ndarray) -> np.ndarray:

        if len(self.context_history) == 0:
            return np.random.randint(self.n_arms, size=len(context))

        stats = self.get_statistics(context)
        if self.method == "ucb":
            arm_idx = self.ucb(stats)
        elif self.method == "thompson":
            arm_idx = self.thompson(stats)
        else:
            ValueError("Unknown method")

        return arm_idx


