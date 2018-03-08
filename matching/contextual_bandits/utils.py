import warnings
from itertools import product
from time import time
from typing import List, Dict, Union, Tuple

import numpy as np
import pandas as pd


def clock_seed():
    """ Simple utility to generate random seed based on time """
    return int(time() * 1e8 % 1e8)


def categorical_dataframe(category_config: Dict):
    """
    Creates an empty pandas DataFrame from category information

    >>> category_config = {'treat1': ['A', 'B'], "treat2": ["C", "D", "E"]}
    >>> df = categorical_dataframe(category_config)
    >>> df
    Empty DataFrame
    Columns: [treat1, treat2]
    Index: []

    >>> df['treat1']
    Series([], Name: treat1, dtype: category
    Categories (2, object): [A, B])

    >>> df['treat2']
    Series([], Name: treat2, dtype: category
    Categories (3, object): [C, D, E])

    Parameters
    ----------
    config: Dict
        Contains

    Returns
    -------

    """

    return pd.DataFrame(
        {key: pd.Categorical([], categories=values) for key, values in category_config.items()}
    )


def get_arm_id_to_integer_mapping(arm_config: List):
    """
    Maps map id back to integer levels

    Example: arm_config = [3,2,3]
        Total number of arms: 3*2*3
            0: [1,0,0, 1,0, 1,0,0] --> (0,0,0)
            1: [1,0,0, 1,0, 0,1,0] --> (0,0,1)
            2: [1,0,0, 1,0, 0,0,1] --> (0,0,2)
            3: [0,0,0, 0,1, 1,0,0] --> (0,1,0)
            etc.

    Parameters
    ----------
    arm_config

    Returns
    -------

    """
    if isinstance(arm_config, Dict):
        arm_config = list(arm_config.values())
    arm_levels = product(*[range(v) for v in arm_config])
    arm_level_count = np.prod(arm_config)
    return dict(zip(range(arm_level_count), arm_levels))


def get_arm_count(arm_config: Union[List, Dict]):
    if isinstance(arm_config, List):
        return np.prod(arm_config)
    elif isinstance(arm_config, Dict):
        return np.prod(list(arm_config.values()))


def cartesian_product(a: np.ndarray, b: np.ndarray) -> (np.ndarray, np.ndarray):
    if len(a.shape) == 0:
        a = a.reshape(-1, 1)
    if len(b.shape) == 0:
        b = b.reshape(-1, 1)
    n_a, n_b = a.shape[0], b.shape[0]
    tiled_a = np.tile(a, (n_b, 1))
    tiled_b = b[np.tile(range(n_b), n_a)]
    return tiled_a, tiled_b


def build_arms(arm_config) -> np.ndarray:
    """
    Builds arm factorial design
    {"arm1":3, "arm2":2} will produce a 6x5 matrix with all combinations

    Returns
    -------
    numpy array of arms

    """
    if isinstance(arm_config, Dict):
        arm_config = list(arm_config.values())
    arm_levels = [np.eye(v) for v in arm_config]
    arms = []
    for combo in product(*arm_levels):
        arms.append(np.hstack(combo))
    return np.array(arms)


def make_covariance_matrix(sigma: float, rho: float, n: int) -> np.ndarray:
    """
    Creates an (n, n) covariance matrix with sigma**2 on diagonal
        and rho*sigma**2 on the off-diagonal

    Parameters
    ----------
    sigma: float
        covariance between two entries
    rho: float
        correlation between two entries
    n: int
        Size of covariance matrix

    Returns
    -------
    cov : np.ndarray
        Covariance matrix

    Examples
    -------

    >>> make_covariance_matrix(1, 0.5, 3)
    array([[1. , 0.5, 0.5],
           [0.5, 1. , 0.5],
           [0.5, 0.5, 1. ]])

    >>> make_covariance_matrix(2, 0.5, 2)
    array([[4., 2.],
           [2., 4.]])

    """

    if abs(rho) > 1:
        ValueError("rho is a correlation, so we need |rho| < 1")

    sigmasq = sigma ** 2
    cov = np.full(shape=(n, n), fill_value=sigmasq * rho)
    cov[np.arange(n), np.arange(n)] = sigmasq
    return cov


def crossjoin(df1: pd.DataFrame,
              df2: pd.DataFrame,
              drop_index: bool = True,
              split=False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Source: https://mkonrad.net/2016/04/16/cross-join--cartesian-product-between-pandas-dataframes.html

    Make a cross join (cartesian product) between two dataframes by using a constant temporary key.
    Also sets a MultiIndex which is the cartesian product of the indices of the input dataframes.
    See: https://github.com/pydata/pandas/issues/5401

    Parameters
    ---------
    df1, df2 : pandas  DataFrames
        dataframes to be merged
    kwargs : dict
        keyword arguments that will be passed to pd.merge

    Returns
    -------
        cross join of df1 and df2
        if split, output is a tuple

    Raises
    ------
        ValueError if df1 and df2 have columns with same name

    Examples
    --------

    >>> df1 = pd.DataFrame([["a","b"],["c","d"]], columns=["A", "B"])
    >>> df2 = pd.DataFrame([[1, 2],[3, 4]], columns=["C", "D"])
    >>> crossjoin(df1, df2)
         A  B  C  D
    0 0  a  b  1  2
      1  a  b  3  4
    1 0  c  d  1  2
      1  c  d  3  4

    """

    if len(df1.columns.intersection(df2.columns)) > 0:
        raise ValueError("DataFrames cannot have same column names")

    df1['_tmpkey'] = 1.0
    df2['_tmpkey'] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        merged = pd.merge(df1, df2, on='_tmpkey').drop('_tmpkey', axis=1)
        merged.index = pd.MultiIndex.from_product((df1.index, df2.index))

    df1.drop('_tmpkey', axis=1, inplace=True)
    df2.drop('_tmpkey', axis=1, inplace=True)

    if drop_index:
        merged.reset_index(drop=True)

    if split:
        return merged[df1.columns], merged[df2.columns]
    else:
        return merged


if __name__ == "__main__":
    import doctest

    doctest.testmod()
