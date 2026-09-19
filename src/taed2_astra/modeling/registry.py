"""Model registry: turns a candidate name from params.yaml into an unfitted model.

Every candidate is Pipeline(preprocessor, estimator), so benchmarking, training
and serving all load one object. Ensembles wrap their members as bare estimators
behind a single shared preprocessor, so preprocessing runs once per request.
"""

from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from taed2_astra.features.build_features import build_preprocessor


class SubsampleClassifier(ClassifierMixin, BaseEstimator):
    """Fit the wrapped classifier on at most ``max_rows`` stratified rows.

    In-context learners such as TabPFN attend over the whole training set at
    prediction time, so their context is bounded. Making the cap explicit lets
    the other candidates train on every row while this one sees a fair sample.
    """

    def __init__(self, estimator: BaseEstimator, max_rows: int = 10_000, random_state: int | None = None) -> None:
        """Store the wrapped estimator and the row cap; nothing is fitted here."""
        self.estimator = estimator
        self.max_rows = max_rows
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SubsampleClassifier":
        """Fit on a stratified subsample and return self."""
        X, y = np.asarray(X), np.asarray(y)
        if len(y) > self.max_rows:
            X, _, y, _ = train_test_split(X, y, train_size=self.max_rows, stratify=y, random_state=self.random_state)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels from the wrapped classifier."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from the wrapped classifier."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


def _logistic_regression(hyperparams: dict, random_state: int) -> BaseEstimator:
    return LogisticRegression(random_state=random_state, **hyperparams)


def _hist_gradient_boosting(hyperparams: dict, random_state: int) -> BaseEstimator:
    return HistGradientBoostingClassifier(random_state=random_state, **hyperparams)


def _tabpfn(hyperparams: dict, random_state: int) -> BaseEstimator:
    try:
        from tabpfn import TabPFNClassifier  # noqa: PLC0415 - optional, heavy dependency
    except ImportError as error:
        raise ImportError("TabPFN is optional. Install it with `uv sync --group foundation`.") from error
    hyperparams = dict(hyperparams)
    max_rows = hyperparams.pop("max_train_rows")
    return SubsampleClassifier(
        TabPFNClassifier(random_state=random_state, **hyperparams),
        max_rows=max_rows,
        random_state=random_state,
    )


def _soft_voting(spec: dict, train_params: dict) -> BaseEstimator:
    members = [(name, build_estimator(name, train_params)) for name in spec["members"]]
    return VotingClassifier(members, voting="soft", weights=spec.get("weights"))


def _stacking(spec: dict, train_params: dict) -> BaseEstimator:
    members = [(name, build_estimator(name, train_params)) for name in spec["members"]]
    return StackingClassifier(
        members,
        final_estimator=LogisticRegression(random_state=train_params["random_state"]),
        cv=spec.get("cv", 3),
        stack_method="predict_proba",
    )


_ESTIMATORS: dict[str, Callable[[dict, int], BaseEstimator]] = {
    "logistic_regression": _logistic_regression,
    "hist_gradient_boosting": _hist_gradient_boosting,
    "tabpfn": _tabpfn,
}
_ENSEMBLES: dict[str, Callable[[dict, dict], BaseEstimator]] = {
    "soft_voting": _soft_voting,
    "stacking": _stacking,
}


def is_ensemble(name: str, train_params: dict) -> bool:
    """Return whether a candidate combines other candidates."""
    return train_params["candidates"][name]["estimator"] in _ENSEMBLES


def build_estimator(name: str, train_params: dict) -> BaseEstimator:
    """Return the unfitted estimator for a candidate, without preprocessing."""
    spec = dict(train_params["candidates"][name])
    kind = spec.pop("estimator")
    if kind in _ENSEMBLES:
        return _ENSEMBLES[kind](spec, train_params)
    if kind in _ESTIMATORS:
        return _ESTIMATORS[kind](spec, train_params["random_state"])
    raise KeyError(f"Unknown estimator '{kind}' for candidate '{name}'. Known: {sorted(_ESTIMATORS | _ENSEMBLES)}")


def build_model(name: str, train_params: dict) -> Pipeline:
    """Return the unfitted preprocessing + estimator pipeline for a candidate."""
    return Pipeline([("preprocessor", build_preprocessor()), ("estimator", build_estimator(name, train_params))])
