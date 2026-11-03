"""下游浅层模型、验证集调参与评估指标。

调参口径：只在 validation 上选超参数，GridSearchCV 的 refit 用 train+validation
重训最终模型，test 只用于最终报告，不参与任何选择。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import LinearSVC, LinearSVR

MODEL_NAMES: dict[str, tuple[str, ...]] = {
    "classification": ("linear_svc", "logistic_regression"),
    "regression": ("ridge", "linear_svr"),
}

DEFAULT_GRIDS: dict[tuple[str, str], dict[str, list[float]]] = {
    ("classification", "linear_svc"): {"C": [0.01, 0.1, 1.0, 10.0]},
    ("classification", "logistic_regression"): {"C": [0.1, 1.0, 10.0]},
    ("regression", "ridge"): {"alpha": [0.1, 1.0, 10.0, 100.0]},
    ("regression", "linear_svr"): {"C": [0.01, 0.1, 1.0, 10.0]},
}

DEFAULT_SCORING: dict[str, str] = {
    "classification": "f1_macro",
    "regression": "neg_root_mean_squared_error",
}


@dataclass
class TrainingResult:
    estimator: Any
    best_params: dict[str, Any]
    candidates: list[tuple[dict[str, Any], float]]


def build_estimator(task: str, name: str, seed: int) -> Any:
    if task == "classification":
        if name == "linear_svc":
            return LinearSVC(
                class_weight="balanced", max_iter=20_000, random_state=seed
            )
        if name == "logistic_regression":
            return LogisticRegression(
                max_iter=2_000, class_weight="balanced", random_state=seed
            )
    else:
        if name == "ridge":
            return Ridge()
        if name == "linear_svr":
            return LinearSVR(max_iter=20_000, random_state=seed)
    raise ValueError(f"未知模型 {name!r}；{task} 可选：{list(MODEL_NAMES[task])}")


def train_model(
    task: str,
    name: str,
    param_grid: dict[str, list[Any]] | None,
    scoring: str | None,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> TrainingResult:
    grid = param_grid or DEFAULT_GRIDS[(task, name)]
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)

    # PredefinedSplit：train 段标 -1 不参与验证，validation 段作为唯一验证折。
    fold_ids = np.full(len(X_all), -1, dtype=np.int64)
    fold_ids[len(X_train) :] = 0

    search: GridSearchCV = GridSearchCV(
        estimator=build_estimator(task, name, seed),
        param_grid=grid,
        scoring=scoring or DEFAULT_SCORING[task],
        cv=PredefinedSplit(fold_ids),
        refit=True,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(X_all, y_all)

    candidates = [
        (dict(params), float(score))
        for params, score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            strict=True,
        )
    ]
    return TrainingResult(
        estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        candidates=candidates,
    )


def evaluate(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if task == "classification":
        labels = sorted(
            {int(v) for v in np.unique(y_true)} | {int(v) for v in np.unique(y_pred)}
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(
                f1_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                )
            ),
            "f1_weighted": float(
                f1_score(
                    y_true, y_pred, labels=labels, average="weighted", zero_division=0
                )
            ),
            "precision_macro": float(
                precision_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                )
            ),
            "recall_macro": float(
                recall_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                )
            ),
            "labels": labels,
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=labels
            ).tolist(),
        }

    pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "r2": float(r2_score(y_true, pred)),
        "rmse": float(root_mean_squared_error(y_true, pred)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "pearson": _corr(y_true, pred, "pearson"),
        "spearman": _corr(y_true, pred, "spearman"),
    }


def decision_scores(
    estimator: Any, X: np.ndarray, pred: np.ndarray
) -> np.ndarray | None:
    """分类取 decision_function；多分类时取预测类对应的那一列。回归模型没有该输出。"""
    if not hasattr(estimator, "decision_function"):
        return None
    scores = np.asarray(estimator.decision_function(X), dtype=np.float64)
    if scores.ndim == 1:
        return scores
    return scores[np.arange(len(pred)), pred.astype(np.int64)]


def _corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    func = pearsonr if method == "pearson" else spearmanr
    return float(func(y_true, y_pred)[0])
