"""Feature preparation shared by training and serving.

Both paths call these functions, so whatever is implemented here is applied
identically at training and at prediction time.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_xy(df: pd.DataFrame, dataset: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into the feature matrix and the label column.

    The label and the grouping column (if any) are never features.
    """
    excluded = [dataset["target"]]
    if dataset.get("group"):
        excluded.append(dataset["group"])
    return df.drop(columns=excluded), df[dataset["target"]]


def build_preprocessor() -> ColumnTransformer:
    """Return the preprocessing steps applied before the estimator.

    Columns are selected by dtype, not by name, so the same code serves any
    dataset described in params.yaml. Missingness indicators are kept because a
    measurement being absent is itself a signal in clinical data. Returning the
    transformer inside the model keeps preprocessing out of the API.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include=np.number)),
            ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
