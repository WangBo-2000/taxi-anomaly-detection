#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train and evaluate anomaly detection models with stratified K-fold
cross validation and manual hyperparameter tuning.

Models:
    - Decision Tree classifier
    - SVM (RBF kernel)
    - Isolation Forest

The script:
    1. Loads a labeled dataset from CSV.
    2. Builds a preprocessing pipeline for continuous and discrete features.
    3. Defines model families and hyperparameter grids.
    4. For each parameter combination, runs stratified K-fold cross validation
       (with metrics computed in the same way as in 3.AblationOneFeature.py).
    5. Uses PR-AUC (average precision) to select the best parameter
       combination for each model family.
    6. Saves a summary table (one row per parameter combination) to CSV,
       including a flag indicating the best setting per model.
"""

import os
import json
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_random_state
from sklearn.base import clone
from itertools import product


RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Set global random seed for reproducibility."""
    np.random.seed(seed)
    _ = check_random_state(seed)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the cleaned taxi order dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def build_preprocessor(
    continuous_features: list[str],
    discrete_features: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer for continuous & discrete features."""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, continuous_features),
            ("cat", categorical_transformer, discrete_features),
        ]
    )
    return preprocessor


def build_models(preprocessor: ColumnTransformer) -> dict:
    """Define the model families and their hyperparameter grids."""
    models: dict[str, dict] = {}

    # Decision Tree
    dt_clf = DecisionTreeClassifier(random_state=RANDOM_SEED)
    dt_pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", dt_clf),
        ]
    )
    dt_param_grid = {
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 5, 10],
        "clf__min_samples_split": [2, 10, 20],
    }
    models["DecisionTree"] = {
        "pipeline": dt_pipeline,
        "param_grid": dt_param_grid,
    }

    # SVM with RBF kernel
    svm_clf = SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_SEED,
    )
    svm_pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", svm_clf),
        ]
    )
    svm_param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", "auto"],
    }
    models["SVM"] = {
        "pipeline": svm_pipeline,
        "param_grid": svm_param_grid,
    }

    # Isolation Forest (unsupervised, treated as anomaly detector)
    iso_clf = IsolationForest(
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso_pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", iso_clf),
        ]
    )
    iso_param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_samples": [0.5, 1.0],
        "clf__contamination": [0.03, 0.05, 0.08],
    }
    models["IsolationForest"] = {
        "pipeline": iso_pipeline,
        "param_grid": iso_param_grid,
    }

    return models


def evaluate_model_with_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    base_pipeline: Pipeline,
    params: dict,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Evaluate one parameter setting with stratified K-fold CV.

    Metrics are computed fold-by-fold in the same way as in
    3.AblationOneFeature.py: accuracy, precision, recall, F1, ROC-AUC,
    and PR-AUC (average precision).
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    acc_list: list[float] = []
    prec_list: list[float] = []
    rec_list: list[float] = []
    f1_list: list[float] = []
    roc_list: list[float] = []
    pr_list: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone the pipeline and set hyperparameters for this fold
        pipe = clone(base_pipeline)
        pipe.set_params(**params)

        pipe.fit(X_train, y_train)

        if model_name == "IsolationForest":
            # IsolationForest: scores < 0 are more likely to be anomalies.
            # We take negative decision_function so larger scores = more anomalous.
            scores = -pipe.decision_function(X_val)
            thr = np.median(scores)
            y_pred = (scores >= thr).astype(int)
            y_score = scores
        else:
            y_pred = pipe.predict(X_val)
            if hasattr(pipe, "predict_proba"):
                y_score = pipe.predict_proba(X_val)[:, 1]
            else:
                # Fallback to decision_function if predict_proba is unavailable
                y_score = pipe.decision_function(X_val)

        acc_list.append(accuracy_score(y_val, y_pred))
        prec_list.append(precision_score(y_val, y_pred, zero_division=0))
        rec_list.append(recall_score(y_val, y_pred, zero_division=0))
        f1_list.append(f1_score(y_val, y_pred, zero_division=0))
        try:
            roc_val = roc_auc_score(y_val, y_score)
        except ValueError:
            roc_val = np.nan
        roc_list.append(roc_val)
        pr_list.append(average_precision_score(y_val, y_score))

        print(
            f"[{model_name}] Fold {fold_idx}: "
            f"Acc={acc_list[-1]:.3f}, Prec={prec_list[-1]:.3f}, "
            f"Rec={rec_list[-1]:.3f}, F1={f1_list[-1]:.3f}, "
            f"ROC-AUC={roc_list[-1]:.3f}, PR-AUC={pr_list[-1]:.3f}"
        )

    # Use ddof=1 for an unbiased estimate of standard deviation
    def mean_std(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1))

    acc_mean, acc_std = mean_std(acc_list)
    prec_mean, prec_std = mean_std(prec_list)
    rec_mean, rec_std = mean_std(rec_list)
    f1_mean, f1_std = mean_std(f1_list)
    roc_mean, roc_std = mean_std(roc_list)
    pr_mean, pr_std = mean_std(pr_list)

    return {
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "precision_mean": prec_mean,
        "precision_std": prec_std,
        "recall_mean": rec_mean,
        "recall_std": rec_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "roc_auc_mean": roc_mean,
        "roc_auc_std": roc_std,
        "pr_auc_mean": pr_mean,
        "pr_auc_std": pr_std,
    }


def grid_search_with_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    base_pipeline: Pipeline,
    param_grid: dict,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Enumerate all parameter combinations and evaluate with K-fold CV.

    PR-AUC (pr_auc_mean) is used as the selection criterion.
    Returns a DataFrame with one row per parameter combination and a
    boolean column `is_best` indicating the best setting.
    """
    if not param_grid:
        raise ValueError(f"Empty param_grid for model {model_name}")

    param_names = sorted(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    rows: list[dict] = []
    best_idx: Optional[int] = None
    best_pr_auc: float = -np.inf

    for values in product(*param_values):
        params = dict(zip(param_names, values))
        print(f"\n=== {model_name}: evaluating params {params} ===")

        metrics = evaluate_model_with_cv(
            X=X,
            y=y,
            model_name=model_name,
            base_pipeline=base_pipeline,
            params=params,
            n_splits=n_splits,
            random_state=random_state,
        )
        metrics["model"] = model_name
        metrics["params"] = json.dumps(params)

        rows.append(metrics)

        if metrics["pr_auc_mean"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc_mean"]
            best_idx = len(rows) - 1

    df = pd.DataFrame(rows)
    df["is_best"] = False
    if best_idx is not None and 0 <= best_idx < len(df):
        df.loc[best_idx, "is_best"] = True

    return df


def main() -> None:
    set_global_seed(RANDOM_SEED)

    # Paths can be changed to your own
    data_path = os.path.join("data", "processed", "taxi_orders_with_labels.csv")
    out_csv = os.path.join("results", "cv_metrics", "model_cv_summary.csv")

    # Feature configuration (adapt to your own columns)
    continuous_features = [
        "DEP_LON",
        "DEP_LAT",
        "DEST_LON",
        "DEST_LAT",
        "Manhattan",
        "DRIVE_MILE",
        "DRIVE_TIME",
        "PRICE",
        "operateSpeed",
    ]

    discrete_features = [
        "isLong",
        "isTime",
        "isPeakTime",
        "isWeekend",
    ]

    label_col = "IS_ANOMALY"

    # 1. Load data
    df = load_data(data_path)

    # 2. Recompute operateSpeed and isPeakTime to ensure consistency
    drive_mile = df["DRIVE_MILE"].astype(float)
    drive_time = df["DRIVE_TIME"].astype(float)

    df["operateSpeed"] = np.where(
        drive_time > 0,
        drive_mile / drive_time,
        np.nan,
    )

    hr = df["hr"].astype(int)
    df["isPeakTime"] = np.where(
        ((hr >= 7) & (hr <= 10)) | ((hr >= 16) & (hr <= 19)),
        1,
        0,
    )

    # 3. Drop rows with missing values in any used feature or label
    df = df.dropna(subset=continuous_features + discrete_features + [label_col]).copy()

    # 4. Prepare feature matrix and labels
    X = df[continuous_features + discrete_features].copy()
    y = df[label_col].astype(int).to_numpy()

    # 5. Build preprocessor and models
    preprocessor = build_preprocessor(
        continuous_features=continuous_features,
        discrete_features=discrete_features,
    )
    models = build_models(preprocessor)

    # 6. For each model, perform grid search using 5-fold CV
    all_results: list[pd.DataFrame] = []
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    for model_name, cfg in models.items():
        print("\n#############################")
        print(f"Grid search for model: {model_name}")
        print("#############################\n")

        df_model = grid_search_with_cv(
            X=X,
            y=y,
            model_name=model_name,
            base_pipeline=cfg["pipeline"],
            param_grid=cfg["param_grid"],
            n_splits=5,
            random_state=RANDOM_SEED,
        )
        all_results.append(df_model)

    summary_df = pd.concat(all_results, ignore_index=True)
    summary_df = summary_df[summary_df["is_best"] == True].reset_index(drop=True)
    summary_df.to_csv(out_csv, index=False)

    print(f"\nCross validation + grid search summary saved to: {out_csv}\n")
    # print(summary_df)


if __name__ == "__main__":
    main()
