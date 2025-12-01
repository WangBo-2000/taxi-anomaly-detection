# -*- coding: utf-8 -*-
"""
Visualization script for the selected anomaly detection model (SVM)
and baselines (Decision Tree, Isolation Forest).

Style requirements:
- Figure width = 6.4 inches
- Figure height = 6.4 * 0.618 inches
- Font family = Times New Roman
- Font size = 11 pt

Inputs:
    - 20230325_test_set_with_evt_anomaly_5pct.csv
    - anomaly_global_bestparams_with_metrics.csv

Outputs:
    Several PNG figures saved to the current directory.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)

# =========================
# 0. Matplotlib style
# =========================

FIG_W = 6.4
FIG_H = 6.4 * 0.618  # ~3.95

plt.rcParams.update({
    "figure.figsize": (FIG_W, FIG_H),
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

# =========================
# 1. Config
# =========================

INPUT_CSV = os.path.join("data", "processed", "taxi_orders_with_labels.csv")
METRICS_CSV = os.path.join("results", "cv_metrics", "model_cv_summary.csv")

label_col = "IS_ANOMALY"
feature_cols = [
    "DEP_LON", "DEP_LAT", "DEST_LON", "DEST_LAT",
    "hr", "isLong", "isTime",
    "Manhattan", "DRIVE_MILE", "DRIVE_TIME", "PRICE", "isWeekend"
]

continuous_cols = [
    "DEP_LON", "DEP_LAT", "DEST_LON", "DEST_LAT",
    "Manhattan", "DRIVE_MILE", "DRIVE_TIME", "PRICE",
]
discrete_cols   = ["hr", "isLong", "isTime", "isWeekend"]

RANDOM_STATE = 42

# =========================
# 2. Load data
# =========================

df = pd.read_csv(INPUT_CSV)

# --- 2.1 Compute operateSpeed = DRIVE_MILE / DRIVE_TIME ---
# Avoid division by zero: if travel time is not positive, set NaN
drive_mile = df["DRIVE_MILE"].astype(float)
drive_time = df["DRIVE_TIME"].astype(float)

df = df.dropna(subset=feature_cols + [label_col]).copy()

X = df[feature_cols].copy()
y = df[label_col].astype(int).values

print(f"Total samples: {len(df)}")

# =========================
# 3. Preprocess + best models
# =========================

preprocess = ColumnTransformer(
    transformers=[
        ("cont", StandardScaler(), continuous_cols),
        ("disc", OneHotEncoder(handle_unknown="ignore"), discrete_cols),
    ]
)

dt_best = DecisionTreeClassifier(
    max_depth=10,
    min_samples_leaf=10,
    min_samples_split=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

svm_best = SVC(
    kernel="rbf",
    C=10.0,
    gamma="auto",
    probability=True,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

if_best = IsolationForest(
    n_estimators=200,
    max_samples=0.5,
    contamination=0.03,
    random_state=RANDOM_STATE,
)

pipelines = {
    "Decision Tree": Pipeline([
        ("preprocess", preprocess),
        ("clf", dt_best),
    ]),
    "SVM": Pipeline([
        ("preprocess", preprocess),
        ("clf", svm_best),
    ]),
    "Isolation Forest": Pipeline([
        ("preprocess", preprocess),
        ("clf", if_best),
    ]),
}

# =========================
# 4. Fit models and get predictions & scores
# =========================

y_pred_dict   = {}
y_score_dict  = {}
roc_auc_dict  = {}

for name, pipe in pipelines.items():
    print(f"Fitting {name} on full data...")
    pipe.fit(X, y)

    if name in ["Decision Tree", "SVM"]:
        y_pred = pipe.predict(X)
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_score = pipe.predict_proba(X)[:, 1]
        elif hasattr(pipe.named_steps["clf"], "decision_function"):
            y_score = pipe.decision_function(X)
        else:
            y_score = y_pred.astype(float)
    else:
        # Isolation Forest: 1 (normal), -1 (anomaly)
        y_if_raw = pipe.predict(X)
        y_pred = (y_if_raw == -1).astype(int)
        # Use pipeline decision_function so that preprocessing is applied
        if hasattr(pipe, "decision_function"):
            # decision_function: higher = more normal, so take negative
            y_score = -pipe.decision_function(X)
        else:
            y_score = y_pred.astype(float)

    y_pred_dict[name] = y_pred
    y_score_dict[name] = y_score

    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc_dict[name] = auc(fpr, tpr)

# =========================
# 5. Bar plot of key CV metrics
# =========================

metrics_df = pd.read_csv(METRICS_CSV)
metrics_df = metrics_df[["model", "f1_mean", "recall_mean", "pr_auc_mean"]]

model_order = ["DecisionTree", "SVM", "IsolationForest"]
display_names = ["Decision Tree", "SVM", "Isolation Forest"]

metrics_df["model"] = pd.Categorical(
    metrics_df["model"],
    categories=model_order,
    ordered=True,
)
metrics_df = metrics_df.sort_values("model")

indicators = ["f1_mean", "recall_mean", "pr_auc_mean"]
indicator_labels = ["F1-score", "Recall", "PR-AUC"]

x = np.arange(len(display_names))
width = 0.25

plt.figure(figsize=(FIG_W, FIG_H))
for i, (metric, label) in enumerate(zip(indicators, indicator_labels)):
    vals = metrics_df[metric].values
    plt.bar(x + (i - 1) * width, vals, width, label=label)

plt.xticks(x, display_names, rotation=0)
plt.ylabel("Score")
plt.ylim(0.0, 1.0)
plt.title("Comparison of key CV metrics")
plt.legend()
plt.tight_layout()
# plt.savefig("results/fig_bar_metrics_cv.png", dpi=500)
plt.close()

# =========================
# 6. Confusion matrix for SVM
# =========================

y_pred_svm = y_pred_dict["SVM"]
cm = confusion_matrix(y, y_pred_svm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal (0)", "Anomaly (1)"],
)

plt.figure(figsize=(FIG_W, FIG_H))
disp.plot(values_format="d", cmap="Blues")
plt.title("Confusion matrix of SVM (full-data fit)")
plt.tight_layout()
plt.savefig("results/fig_svm_confusion_matrix.png", dpi=500)
plt.close()

# =========================
# 7. ROC curves
# =========================

plt.figure(figsize=(FIG_W, FIG_H))
for name in pipelines.keys():
    y_score = y_score_dict[name]
    fpr, tpr, _ = roc_curve(y, y_score)
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves (full-data fit)")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig_roc_curves.png", dpi=500)
plt.close()

# =========================
# 8. Precision–Recall curves
# =========================

plt.figure(figsize=(FIG_W, FIG_H))
for name in pipelines.keys():
    y_score = y_score_dict[name]
    precision, recall, _ = precision_recall_curve(y, y_score)
    plt.plot(recall, precision, label=name)

baseline = y.mean()
plt.hlines(baseline, 0, 1, linestyles="--", color="gray")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall curves (full-data fit)")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig_pr_curves.png", dpi=500)
plt.close()

# =========================
# 9. Threshold analysis for SVM
# =========================

y_score_svm = y_score_dict["SVM"]
precision, recall, thresholds = precision_recall_curve(y, y_score_svm)

thresholds_plot = thresholds
precision_plot = precision[1:]
recall_plot = recall[1:]
f1_plot = 2 * precision_plot * recall_plot / (precision_plot + recall_plot + 1e-8)

plt.figure(figsize=(FIG_W, FIG_H))
plt.plot(thresholds_plot, precision_plot, label="Precision")
plt.plot(thresholds_plot, recall_plot, label="Recall")
plt.plot(thresholds_plot, f1_plot, label="F1-score")

best_idx = np.argmax(f1_plot)
best_thr = thresholds_plot[best_idx]
best_f1 = f1_plot[best_idx]
plt.axvline(best_thr, linestyle="--", color="gray")
plt.text(
    best_thr, best_f1,
    f"best F1={best_f1:.3f}\nthr={best_thr:.3f}",
    va="bottom", ha="left",
)

plt.xlabel("Decision threshold on SVM score")
plt.ylabel("Metric value")
plt.title("SVM Precision/Recall/F1 vs threshold")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig_svm_threshold_curve.png", dpi=500)
plt.close()

# =========================
# 10. Score distribution for SVM
# =========================

plt.figure(figsize=(FIG_W, FIG_H))
scores_normal  = y_score_svm[y == 0]
scores_anomaly = y_score_svm[y == 1]

plt.hist(scores_normal, bins=40, alpha=0.6, label="Normal (y=0)")
plt.hist(scores_anomaly, bins=40, alpha=0.6, label="Anomaly (y=1)")

plt.xlabel("SVM anomaly score")
plt.ylabel("Count")
plt.title("Distribution of SVM scores")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig_svm_score_distribution.png", dpi=500)
plt.close()

# =========================
# 11. Top-K Precision / Recall for SVM vs Isolation Forest
# =========================

def topk_curve(y_true, y_score, ks):
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    precision_k = []
    recall_k = []
    total_pos = y_true.sum()

    for k in ks:
        k = min(k, len(y_sorted))
        topk = y_sorted[:k]
        tp_k = topk.sum()
        precision_k.append(tp_k / k if k > 0 else 0.0)
        recall_k.append(tp_k / total_pos if total_pos > 0 else 0.0)

    return np.array(precision_k), np.array(recall_k)


ks = np.arange(100, 2001, 100)

plt.figure(figsize=(FIG_W, FIG_H))
for name in ["SVM", "Isolation Forest"]:
    y_score = y_score_dict[name]
    p_at_k, r_at_k = topk_curve(y, y_score, ks)
    plt.plot(ks, p_at_k, marker="o", label=f"{name}")

plt.xlabel("K (top-K highest-scoring trips)")
plt.ylabel("Precision@K")
plt.title("Top-K Precision (SVM vs Isolation Forest)")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig_topk_precision.png", dpi=500)
plt.close()

plt.figure(figsize=(FIG_W, FIG_H))
for name in ["SVM", "Isolation Forest"]:
    y_score = y_score_dict[name]
    p_at_k, r_at_k = topk_curve(y, y_score, ks)
    plt.plot(ks, r_at_k, marker="o", label=f"{name}")

plt.xlabel("K (top-K highest-scoring trips)")
plt.ylabel("Recall@K")
plt.title("Top-K Recall (SVM vs Isolation Forest)")
plt.legend()
plt.tight_layout()
plt.savefig("results/fig_topk_recall.png", dpi=500)
plt.close()

print(
    "All figures have been saved with size 6.4 x 6.4*0.618 inches "
    "and Times New Roman font (12pt)."
)
