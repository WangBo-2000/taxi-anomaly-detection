# -*- coding: utf-8 -*-
"""
Feature ablation experiments for SVM anomaly detection model.

Base features:
    base_continuous_cols = [
        'DEP_LON', 'DEP_LAT', 'DEST_LON', 'DEST_LAT',
        'Manhattan', 'DRIVE_MILE', 'DRIVE_TIME', 'PRICE', 'operateSpeed'
    ]
    base_discrete_cols   = [
        'hr', 'isLong', 'isTime', 'isPeakTime', 'isWeekend'
    ]

Ablation settings:
    - remove DEP_LON + DEP_LAT + DEST_LON + DEST_LAT (as one group)
    - remove Manhattan
    - remove DRIVE_MILE
    - remove DRIVE_TIME
    - remove PRICE
    - remove operateSpeed
    - remove hr
    - remove isLong
    - remove isTime
    - remove isPeakTime
    - remove isWeekend

StartCode / EndCode are NOT used as features and NOT ablated.
"""
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# =========================
# 0. Config
# =========================

INPUT_CSV = os.path.join("data", "processed", "taxi_orders_with_labels.csv")
OUT_CSV   = "results/anomaly_ablation_svm.csv"

label_col = "IS_ANOMALY"

RANDOM_STATE = 42
N_SPLITS = 5

# =========================
# 1. Load data and build extra features
# =========================

df = pd.read_csv(INPUT_CSV)

# --- 1.1 Compute operateSpeed = DRIVE_MILE / DRIVE_TIME ---
# Avoid division by zero: if travel time is not positive, set NaN
drive_mile = df["DRIVE_MILE"].astype(float)
drive_time = df["DRIVE_TIME"].astype(float)

df["operateSpeed"] = np.where(
    drive_time > 0,
    drive_mile / drive_time,
    np.nan,
)

# --- 1.2 Construct isPeakTime flag ---
# Based on hour of day: 1 if hr in [7, 10] or [16, 19], otherwise 0
# Assume df["hr"] is already an integer hour between 0 and 23
hr = df["hr"].astype(int)
df["isPeakTime"] = np.where(
    ((hr >= 7) & (hr <= 10)) | ((hr >= 16) & (hr <= 19)),
    1,
    0,
)

# =========================
# 2. Define base feature sets
# =========================

base_continuous_cols = [
    "DEP_LON", "DEP_LAT", "DEST_LON", "DEST_LAT",
    "Manhattan", "DRIVE_MILE", "DRIVE_TIME", "PRICE", "operateSpeed",
]

# StartCode and EndCode are no longer used as features
base_discrete_cols = ["isLong", "isTime", "isPeakTime", "isWeekend"]

feature_cols_all = base_continuous_cols + base_discrete_cols

# Drop samples that have NaN in any selected feature or in the label
df = df.dropna(subset=feature_cols_all + [label_col]).copy()

y = df[label_col].astype(int).values
# print(f"Total samples after cleaning: {len(df)}, anomaly ratio: {y.mean():.4f}")

# =========================
# 3. Define ablation experiments
# =========================

ablation_settings = {
    "remove operateSpeed and isPeakTime": {
        "remove_cont": ["operateSpeed"],
        "remove_disc": ['isPeakTime'],
        "description": "remove operateSpeed and isPeakTime",
    },
    "None": {
        "remove_cont": [],
        "remove_disc": [],
        "description": "remove nothing",
    },
}

# =========================
# 4. Helper: build SVM pipeline
# =========================

def build_pipeline(continuous_cols, discrete_cols):
    """
    Build preprocessing + SVM pipeline for given feature subsets.
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("cont", StandardScaler(), continuous_cols),
            ("disc", OneHotEncoder(handle_unknown="ignore"), discrete_cols),
        ]
    )

    svm_best = SVC(
        kernel="rbf",
        C=10.0,
        gamma="auto",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", svm_best),
    ])
    return pipe

# =========================
# 5. Run ablation experiments (5-fold CV)
# =========================

results = []
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

for exp_name, cfg in ablation_settings.items():
    # Features kept in the current experiment
    cont_cols = [c for c in base_continuous_cols if c not in cfg["remove_cont"]]
    disc_cols = [c for c in base_discrete_cols if c not in cfg["remove_disc"]]
    feat_cols = cont_cols + disc_cols

    print(f"\n=== Experiment: {exp_name} ===")
    print(f"Description: {cfg['description']}")
    print(f"Continuous features: {cont_cols}")
    print(f"Discrete features:   {disc_cols}")

    # Keep X as a DataFrame so that ColumnTransformer can use column names
    X = df[feat_cols].copy()

    pipe = build_pipeline(continuous_cols=cont_cols, discrete_cols=disc_cols)

    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    roc_list, pr_list = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_val)
        y_score = pipe.predict_proba(X_val)[:, 1]

        acc_list.append(accuracy_score(y_val, y_pred))
        prec_list.append(precision_score(y_val, y_pred, zero_division=0))
        rec_list.append(recall_score(y_val, y_pred, zero_division=0))
        f1_list.append(f1_score(y_val, y_pred, zero_division=0))
        roc_list.append(roc_auc_score(y_val, y_score))
        pr_list.append(average_precision_score(y_val, y_score))

        print(
            f"  Fold {fold_idx}: "
            f"Acc={acc_list[-1]:.3f}, "
            f"Prec={prec_list[-1]:.3f}, "
            f"Rec={rec_list[-1]:.3f}, "
            f"F1={f1_list[-1]:.3f}, "
            f"ROC-AUC={roc_list[-1]:.3f}, "
            f"PR-AUC={pr_list[-1]:.3f}"
        )

    res = {
        "experiment": exp_name,
        "description": cfg["description"],
        "n_cont_features": len(cont_cols),
        "n_disc_features": len(disc_cols),
        "features_used": ";".join(feat_cols),
        "accuracy_mean": np.mean(acc_list),
        "accuracy_std": np.std(acc_list, ddof=1),
        "precision_mean": np.mean(prec_list),
        "precision_std": np.std(prec_list, ddof=1),
        "recall_mean": np.mean(rec_list),
        "recall_std": np.std(rec_list, ddof=1),
        "f1_mean": np.mean(f1_list),
        "f1_std": np.std(f1_list, ddof=1),
        "roc_auc_mean": np.mean(roc_list),
        "roc_auc_std": np.std(roc_list, ddof=1),
        "pr_auc_mean": np.mean(pr_list),
        "pr_auc_std": np.std(pr_list, ddof=1),
    }
    results.append(res)

# =========================
# 6. Save results
# =========================

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nAblation results saved to: {OUT_CSV}")
print(results_df[[
    "experiment", "accuracy_mean", "precision_mean",
    "recall_mean", "f1_mean", "roc_auc_mean", "pr_auc_mean",
]])
