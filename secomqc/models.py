"""Candidate models, all built as leakage-safe pipelines."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .preprocessing import MannWhitneyScreen, SensorFilter


class RandomScorer(BaseEstimator, ClassifierMixin):
    """No-skill reference: a random risk score for every lot."""

    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self._rng = np.random.default_rng(self.random_state)
        return self

    def predict_proba(self, X):
        p = self._rng.random(len(X))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


MODELS = ["random_forest", "gradient_boosting", "logistic", "no_skill"]
DEFAULT_MODEL = "random_forest"


def make_model(name: str = DEFAULT_MODEL, k: int = 50, seed: int = 0) -> Pipeline:
    """Sensor filter -> in-fold screening -> imputation -> classifier."""
    steps = [("filter", SensorFilter()), ("screen", MannWhitneyScreen(k=k)),
             ("impute", SimpleImputer(strategy="median"))]
    if name == "random_forest":
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features=0.3,
                                     class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    elif name == "gradient_boosting":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=4, min_child_samples=15,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
                                 is_unbalance=True, verbose=-1, random_state=seed)
    elif name == "logistic":
        steps.append(("scale", RobustScaler()))
        clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000)
    elif name == "no_skill":
        clf = RandomScorer(random_state=seed)
    else:
        raise ValueError(f"unknown model {name!r}, expected one of {MODELS}")
    pipe = Pipeline(steps + [("model", clf)])
    pipe.set_output(transform="pandas")
    return pipe
