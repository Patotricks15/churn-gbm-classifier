import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from churn.config import RANDOM_STATE
from churn.features import BASELINE_FEATURES, FEATURE_COLUMNS

PARAM_SPACE = {
    "n_estimators": randint(100, 500),
    "learning_rate": loguniform(0.01, 0.2),
    "num_leaves": randint(8, 64),
    "min_child_samples": randint(10, 100),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_lambda": uniform(0.0, 5.0),
}


def scale_pos_weight(y) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())


def fit_baseline_logistic(train: pd.DataFrame):
    model = make_pipeline(
        FunctionTransformer(np.log1p),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(train[BASELINE_FEATURES], train["churned"])
    return model


def predict_baseline(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[BASELINE_FEATURES])[:, 1]


def fit_tuned_lgbm(train: pd.DataFrame, validation: pd.DataFrame, n_iter: int):
    """Search hyperparameters by PR-AUC on the validation snapshot, then refit on train + validation."""
    fit_df = pd.concat([train, validation], ignore_index=True)
    X, y = fit_df[FEATURE_COLUMNS], fit_df["churned"]
    test_fold = np.r_[np.full(len(train), -1), np.zeros(len(validation), dtype=int)]
    search = RandomizedSearchCV(
        estimator=LGBMClassifier(
            random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight(y),
            subsample_freq=1,
            n_jobs=1,
            verbose=-1,
        ),
        param_distributions=PARAM_SPACE,
        n_iter=n_iter,
        scoring="average_precision",
        cv=PredefinedSplit(test_fold),
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def predict_lgbm(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURE_COLUMNS])[:, 1]


def gain_importance(model) -> pd.DataFrame:
    gain = model.booster_.feature_importance(importance_type="gain")
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "gain": gain}).sort_values("gain", ascending=False)
