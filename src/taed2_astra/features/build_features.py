"""Feature preparation shared by training and serving.

Both paths call these functions, so whatever is implemented here is applied
identically at training and at prediction time.

TODO(team): implement both functions.
"""

import pandas as pd
from sklearn.pipeline import Pipeline


def split_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into the feature matrix and the label column."""
    raise NotImplementedError("Select the feature columns and return (X, y)")


def build_preprocessor() -> Pipeline:
    """Return the preprocessing steps applied before the estimator.

    Returning a Pipeline (rather than transforming in place) keeps the
    preprocessing inside the saved model, so serving cannot skip a step.
    """
    raise NotImplementedError("Build the preprocessing Pipeline")
