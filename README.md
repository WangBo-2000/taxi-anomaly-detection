# Taxi Anomaly Detection with SVM and Baselines

This repository contains code for detecting anomalous taxi trips
(for example detours and overcharging) using classical machine learning models.

## Files

- `2.AbnormalitiesDetection.py`  
  Selects global hyperparameters for three models (Decision Tree, SVM, Isolation Forest)
  based on PR-AUC and evaluates them with 5-fold cross validation.

- `3.Visualization.py`  
  Fits the tuned models on the full dataset and produces figures:
  bar plots of key metrics, ROC and PR curves, SVM confusion matrix,
  threshold analysis, score distribution, and top-K precision/recall.

- `4.Ablation.py`  
  Runs feature ablation experiments for the SVM model to study the contribution
  of different feature groups.

## Data

The scripts expect an input CSV:

- `data/processed/taxi_orders_with_labels.csv`

You can either place the full dataset in the `data` directory or
modify the path constants (`INPUT_CSV` variables) in the scripts.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
