#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train and evaluate anomaly detection models with stratified K-fold
cross validation and hyperparameter tuning.

Models:
    - Decision Tree classifier
    - SVM (RBF kernel)
    - Isolation Forest

The script:
    1. Loads a labeled dataset from CSV.
    2. Builds a preprocessing pipeline for continuous and discrete features.
    3. Defines model families and hyperparameter grids.
    4. Runs grid search with stratified K-fold cross validation
       using PR-AUC as the selection metric.
    5. Computes mean and standard deviation of several metrics
       over the outer folds.
    6. Saves a summary table to CSV for later visualization.
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_random_state


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
    """
    Build a ColumnTransformer that standardizes continuous features
    and one-hot encodes discrete features.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
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
    """
    Define the model families and their hyperparameter grids.

    Returns
    -------
    dict
        Mapping from model name to a dictionary with keys:
        - "pipeline": sklearn Pipeline
        - "param_grid": dict of hyperparameters for GridSearchCV
    """
    models = {}

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

    # Isolation Forest (unsupervised)
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


def run_nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    models: dict,
    n_splits: int = 5,
    out_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run outer stratified K-fold cross validation.
    For each outer fold:
        - Run GridSearchCV on the training part using PR-AUC.
        - Evaluate the best model on the validation fold.
    Aggregate metrics over folds for each model family.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    y : ndarray
        Binary labels (1 for anomaly, 0 for normal).
    models : dict
        Output from build_models().
    n_splits : int
        Number of stratified folds.
    out_path : str or None
        If not None, save the summary CSV to this path.

    Returns
    -------
    DataFrame
        Summary metrics for each model family.
    """
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
    )

    rows = []

    for model_name, cfg in models.items():
        print(f"\n=== Model: {model_name} ===")
        pipeline = cfg["pipeline"]
        param_grid = cfg["param_grid"]

        # Record per-fold metrics
        fold_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": [],
            "pr_auc": [],
        }

        best_params_list = []
        best_pr_aucs = []

        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
            print(f"  Fold {fold_idx + 1}/{n_splits}")

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            # Inner tuning using PR-AUC of the positive class
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="average_precision",
                cv=3,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            best_params_list.append(grid.best_params_)
            best_pr_aucs.append(grid.best_score_)

            # Predictions on the validation fold
            if model_name == "IsolationForest":
                # IsolationForest uses -1 for outliers and +1 for inliers
                scores = -best_model.decision_function(X_valid)
                # Convert scores to binary labels using median threshold
                thr = np.median(scores)
                y_pred = (scores >= thr).astype(int)
                y_score = scores
            else:
                y_pred = best_model.predict(X_valid)
                # Use probability of positive class if available
                if hasattr(best_model, "predict_proba"):
                    y_prob = best_model.predict_proba(X_valid)[:, 1]
                    y_score = y_prob
                else:
                    y_dec = best_model.decision_function(X_valid)
                    y_score = y_dec

            # Compute metrics for this fold
            acc = accuracy_score(y_valid, y_pred)
            prec = precision_score(y_valid, y_pred, zero_division=0)
            rec = recall_score(y_valid, y_pred, zero_division=0)
            f1 = f1_score(y_valid, y_pred, zero_division=0)
            try:
                roc = roc_auc_score(y_valid, y_score)
            except ValueError:
                roc = np.nan
            pr_auc = average_precision_score(y_valid, y_score)

            fold_metrics["accuracy"].append(acc)
            fold_metrics["precision"].append(prec)
            fold_metrics["recall"].append(rec)
            fold_metrics["f1"].append(f1)
            fold_metrics["roc_auc"].append(roc)
            fold_metrics["pr_auc"].append(pr_auc)

        # Aggregate metrics over folds
        def mean_std(values: list[float]) -> tuple[float, float]:
            arr = np.asarray(values, dtype=float)
            return float(arr.mean()), float(arr.std())

        acc_mean, acc_std = mean_std(fold_metrics["accuracy"])
        prec_mean, prec_std = mean_std(fold_metrics["precision"])
        rec_mean, rec_std = mean_std(fold_metrics["recall"])
        f1_mean, f1_std = mean_std(fold_metrics["f1"])
        roc_mean, roc_std = mean_std(fold_metrics["roc_auc"])
        pr_mean, pr_std = mean_std(fold_metrics["pr_auc"])

        row = {
            "model": model_name,
            "best_params": json.dumps(best_params_list[-1]),
            "best_cv_pr_auc_for_selection": float(np.mean(best_pr_aucs)),
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
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"\nSaved CV summary to {out_path}")

    return summary_df


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
        "hr",
        "isLong",
        "isTime",
        "isPeakTime",
        "isWeekend",
    ]

    label_col = "IS_ANOMALY"

    df = load_data(data_path)

    drive_mile = df["DRIVE_MILE"].astype(float)
    drive_time = df["DRIVE_TIME"].astype(float)

    df["operateSpeed"] = np.where(
        drive_time > 0,
        drive_mile / drive_time,
        np.nan
    )

    hr = df["hr"].astype(int)
    df["isPeakTime"] = np.where(
        ((hr >= 7) & (hr <= 10)) | ((hr >= 16) & (hr <= 19)),
        1,
        0
    )

    X = df[continuous_features + discrete_features].copy()
    y = df[label_col].astype(int).to_numpy()

    preprocessor = build_preprocessor(
        continuous_features=continuous_features,
        discrete_features=discrete_features,
    )

    models = build_models(preprocessor)
    summary_df = run_nested_cv(
        X=X,
        y=y,
        models=models,
        n_splits=5,
        out_path=out_csv,
    )

    print("\nCross validation summary:")
    print(summary_df)


if __name__ == "__main__":
    main()
