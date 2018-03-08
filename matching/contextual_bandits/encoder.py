from itertools import product
from typing import Union

import numpy as np
import pandas as pd

DataType = Union[np.ndarray, pd.Series, pd.DataFrame]


class CategoricalEncoder:
    """
    Examples
    --------
    `CategoricalEncoder` applies one-hot-encodes
    categorical columns of the DataFrame, but also keeps
    the information so as convert everything back
    to original format later.

    There are three encodings:

    + `category`: exactly as in original data, e.g.
           A      B
        "blue"    0
        "red"     1
        "blue"    2

     + `dummy`: one-hot encoding
         A_blue  A_red    B_0   B_1   B_2
           1       0       1     0     0
           0       1       0     1     0
           1       0       0     0     1

    + `factorial`: integer corresponding to unique combination
            0
            1
            2
            ...


    """

    def __init__(self):
        self.classes = None
        self.dummy_to_factorial_mapping = None
        self.factorial_to_dummy_mapping = None
        self.dummy_combinations = None
        self.class_combinations = None
        self.arm_count = None

    def fit(self, df: pd.DataFrame) -> None:
        self.classes = {column: list(df[column].cat.categories.values) for column in df.columns}

        self.class_combinations = pd.DataFrame(data=list(product(*self.classes.values())),
                                               columns=df.columns) \
            .astype(df.dtypes)

        self.dummy_combinations = pd.get_dummies(self.class_combinations)

        self.factorial_to_dummy_mapping = self.dummy_combinations.apply(tuple, axis=1).to_dict()
        self.dummy_to_factorial_mapping = {v: k for k, v in self.factorial_to_dummy_mapping.items()}
        self.arm_count = self.class_combinations.shape[0]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(df)

    def fit_transform(self, df: pd.DataFrame) -> DataType:
        self.fit(df)
        return self.transform(df)

    def dummy_to_factorial(self, dummies: pd.DataFrame) -> pd.Series:
        return dummies.apply(tuple, axis=1).map(self.dummy_to_factorial_mapping)

    def factorial_to_classes(self, factorial: DataType) -> Union[pd.DataFrame, pd.Series]:
        return self.class_combinations.loc[factorial].reset_index(drop=True)

    def random_dummy(self, batch_size):
        return self.dummy_combinations \
            .sample(batch_size, replace=True) \
            .reset_index(drop=True)

    def random_classes(self, batch_size):
        return self.class_combinations \
            .sample(batch_size, replace=True) \
            .reset_index(drop=True)


if __name__ == "__main__":
    import pickle

    df = pickle.load(open("/Users/vitorhadad/Desktop/test_df.pkl", "rb"))
    enc = CategoricalEncoder()
    original = df.select_dtypes("category")
    enc.fit(original)
    new_classes = enc.random_classes(10)
    new_dummy = enc.transform(new_classes)
    new_factorial = enc.dummy_to_factorial(new_dummy)
