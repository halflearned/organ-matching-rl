#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:39:02 2018
@author:
    Imanol Arrieta Ibarra (imanolarrieta)
    Maria Dimakopoulou (mdimakopoulou)
    Vitor Hadad (halflearned)
"""

from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from matching.contextual_bandits.encoder import CategoricalEncoder
from matching.contextual_bandits.utils import crossjoin
from matching.environment.abo_environment import ABOKidneyExchange
from matching.environment.base_environment import BaseKidneyExchange
from matching.utils.env_utils import snapshot


class Agent:
    """
    Contextual bandit agent

    ----------
    arms
    method
    context_history
    arm_history
    reward_history
    bootstrap_samples
    estimator : BaseEstimator
        Any scikit-learn estimator or pipeline

    """

    def __init__(self,
                 method: str,
                 estimator: BaseEstimator):

        self.arm_encoder = CategoricalEncoder()
        self.arms = None

        self.method = method
        self.estimator = estimator

        self.n = 0

    def __str__(self):
        return "Agent(estimator={}, method={})".format(str(self.estimator), self.method)

    def fit(self,
            context_history: pd.DataFrame,
            arm_feature_history: pd.DataFrame,
            reward_history: pd.DataFrame) -> None:

        if self.n == 0:
            self.arm_encoder.fit(arm_feature_history)

        # If there's not data, just return
        if len(context_history) == 0:
            return

        # Update data size
        self.n += len(context_history)

        # One-hot encoded (i.e., dummies) for arms
        encoded_arm_history = self.arm_encoder.transform(arm_feature_history)

        # Fit on training data and cross-validate if enabled
        context_arm = np.hstack([context_history, encoded_arm_history])
        self.estimator.fit(context_arm, reward_history)

    def get_statistics(self, new_contexts: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        # Predict on training data for all arms
        new_context_arm = crossjoin(new_contexts, self.arm_encoder.dummy_combinations)
        mean_rewards, variance_rewards = self.estimator.predict(new_context_arm)
        # TODO: Profile this vs old for-loop approach

        # Reshape to (batch_size, arm_count)
        mean_rewards = mean_rewards.reshape(-1, self.arm_encoder.arm_count)
        variance_rewards = variance_rewards.reshape(-1, self.arm_encoder.arm_count)

        return mean_rewards, variance_rewards

    def ucb(self,
            mean_rewards: pd.DataFrame,
            variance_rewards: pd.DataFrame) -> np.ndarray:

        """
        Best arm index according to UCB algorithm.
        Computes the score for each arm-context combination,
        then outputs the argmax arm for each context.

        Parameters
        ----------
        mean_rewards : pd.DataFrame
        variance_rewards : pd.DataFrame
            Statistics for each context-arm

        Returns
        -------
        arm_id : np.ndarray
            Index of chosen arm
        """
        score = mean_rewards + np.sqrt(2 * np.log(self.n) * variance_rewards)
        arm_id = np.argmax(score, axis=1)
        return arm_id

    def thompson(self,
                 mean_rewards: pd.DataFrame,
                 variance_rewards: pd.DataFrame) -> np.ndarray:

        """
        Thompson-sampled best arm index.
        Draws a reward from Normal(mu = stats["mean"], sigma = stats["std"])
        for each arm and context, then outputs the argmax arm for each context.

        Parameters
        ----------
        mean_rewards : pd.DataFrame
        variance_rewards : pd.DataFrame
            Statistics for each arm

        Returns
        -------
        arm_id : np.ndarray
            Index of chosen arm
        """

        draws = np.random.normal(mean_rewards, np.sqrt(variance_rewards))
        arm_id = np.argmax(draws, axis=1)
        return arm_id

    def choose_arms(self, context: pd.DataFrame, factorial=False) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Selects best arm for each context according to predicted statistics
        Parameters
        ----------
        context : np.ndarray
            New context for which arms will be chosen.
        factorial : bool
            If False, output has same format as input
            If True, output has unique id coming from factorial design

        Returns
        -------
        chosen : np.ndarray
            Best arm for each row in context.
        """
        if self.n == 0:
            random_arms = self.arm_encoder.random_classes(batch_size=len(context))
            random_arms.index = context.index
            return random_arms

        statistics = self.get_statistics(context)

        if self.method == "ucb":
            arm_id = self.ucb(*statistics)
        elif self.method == "thompson":
            arm_id = self.thompson(*statistics)
        else:
            raise ValueError("Unknown method")

        if factorial:
            return arm_id
        else:
            arm_classes = self.arm_encoder.factorial_to_classes(arm_id)
            arm_classes.index = context.index
            return arm_classes

    def reset(self) -> None:
        """
        For the estimators
        Constructs a new estimator with the same parameters.
        It yields a new estimator with the same parameters that
         has not been fit on any data.

        Also resets the number of observations seen to zero.

        """
        self.estimator.statistical_method = clone(self.estimator.statistical_method)
        self.estimator.models = []

        if self.estimator.preprocessor is not None:
            self.estimator.preprocessor = clone(self.estimator.preprocessor)

        if self.estimator.cross_validation_method is not None:
            self.estimator.cross_validation_method = clone(self.estimator.cross_validation_method)
        self.estimator.parameter_history = defaultdict(list)
        self.n = 0


class BootstrapEstimator(BaseEstimator):

    def __init__(self,
                 statistical_method: BaseEstimator,
                 preprocessor: BaseEstimator = None,
                 cross_validation_method: BaseEstimator = None,
                 cross_validation_parameters: Dict = None,
                 keep_parameters: List = None,
                 bootstrap_sample_count: int = 100):

        self.preprocessor = preprocessor
        self.statistical_method = statistical_method
        self.pipe = Pipeline([("preprocess", self.preprocessor),
                              ("statistical_method", self.statistical_method)])

        self.keep_parameters = keep_parameters if keep_parameters else []
        self.parameter_history = defaultdict(list)
        self.bootstrap_sample_count = bootstrap_sample_count

        self.models = []

        self.cross_validation = cross_validation_method is not None
        if self.cross_validation:
            self.cross_validation_parameters = cross_validation_parameters
            self.cross_validation_method = cross_validation_method

    def __str__(self):
        return "BootstrapEstimator({}, {})".format(
            str(self.statistical_method).split("(")[0],
            str(self.cross_validation_method).split("(")[0]
        )

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> None:
        """
        Fits and performs cross-validation if enabled.

        Parameters
        ----------
        X, y :
            Training data
        """

        if self.cross_validation:
            self.cross_validate(X, y)

        for b in range(self.bootstrap_sample_count):

            # Sample bootstrap indices
            X_bs, y_bs = resample(X, y, random_state=b)

            # Refit on the bootstrap sample
            self.pipe.fit(X_bs, y_bs)

            # Save for prediction later
            self.models.append(deepcopy(self.pipe))  # TODO: Profile. Deepcopy can be *very expensive*

            for param in self.keep_parameters:
                self.parameter_history[param].append(vars(self.statistical_method)[param])

    def cross_validate(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> None:
        """ Modifies the statistical estimator to have cross-validated parameters """
        self.cross_validation_method.fit(X, y)

        # Assign tuned parameters to statistical_method
        for param_name, cv_param_name in self.cross_validation_parameters.items():
            vars(self.statistical_method)[param_name] = vars(self.cross_validation_method)[cv_param_name]

    def predict(self, X: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        predictions = []

        for model in self.models:
            predictions.append(model.predict(X))

        predictions = np.vstack(predictions)

        return np.mean(predictions, axis=0), np.var(predictions, axis=0)


class KidneyBandit:

    def __init__(self,
                 agent: Agent,
                 environment: BaseKidneyExchange,
                 t: int):
        self.agent = agent
        self.env = snapshot(environment, t)
        self.t = t
        self.subg = self.env.subgraph(self.env.get_living(self.t))
        self.edges = self.subg.edges

    def get_initial_matrix(self, ndd_only=False):
        vs, ws = np.array(list(self.edges)).T
        X = self.env.X(self.t, dtype="pandas")
        return np.hstack([X.loc[vs].values,
                          X.loc[ws].values[:, :-2]])

    def testit(self):





if __name__ == "__main__":
    from sklearn.linear_model import Lasso, LassoCV

    # from sklearn.preprocessing import PolynomialFeatures

    abo = ABOKidneyExchange(5, 1, 100, fraction_ndd=0.2)
    estimator = BootstrapEstimator(
        statistical_method=Lasso(fit_intercept=False),
        cross_validation_method=LassoCV(fit_intercept=False),
        preprocessor=None  # PolynomialFeatures(degree=2)
    )

    agent = Agent(method="thompson",
                  estimator=estimator)

    bandit = KidneyBandit(agent=agent,
                          environment=abo,
                          t=10)

    XX = bandit.get_matrix()
