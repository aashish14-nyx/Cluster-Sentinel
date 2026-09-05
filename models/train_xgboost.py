"""Cluster Sentinel - XGBoost with a time-based train/validation/test split.

Uses the canonical causal features from features.csv.
Validation is reserved for model/threshold decisions; the final test set
remains untouched until final evaluation.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

FEATURE_COLS = [
    "velocity_1h",
    "velocity_24h",
    "amount_log_ratio",
    "time_since_last_txn_sec",
    "has_previous_txn",
    "is_new_device",
    "is_new_instrument",
    "device_customer_count_before",
    "instrument_customer_count_before",
]


def time_based_split(df, train_frac=0.70, val_frac=0.15):
    """Chronological 70/15/15 split with deterministic tie ordering."""
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)

    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1.")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def evaluate(y_true, y_pred, y_score, label):
    """Print threshold metrics plus ranking metrics."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    print(f"\n=== {label} ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"PR-AUC:    {pr_auc:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")
    print(f"Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def validate_input(df):
    required = FEATURE_COLS + ["transaction_id", "timestamp", "is_fraud"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[FEATURE_COLS + ["is_fraud"]].isna().any().any():
        raise ValueError("Missing values detected in model inputs.")

    if not df["transaction_id"].is_unique:
        raise ValueError("Duplicate transaction_id values detected.")

    if not df["is_fraud"].isin([0, 1]).all():
        raise ValueError("is_fraud must contain only 0/1 values.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        default="../features/features.csv",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    validate_input(df)

    train, val, test = time_based_split(df)

    for name, part in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        print(
            f"{name}: {len(part):,} rows "
            f"({int(part['is_fraud'].sum())} fraud, "
            f"{part['is_fraud'].mean():.3%})"
        )

    X_train = train[FEATURE_COLS]
    y_train = train["is_fraud"].astype(int)

    X_val = val[FEATURE_COLS]
    y_val = val["is_fraud"].astype(int)

    X_test = test[FEATURE_COLS]
    y_test = test["is_fraud"].astype(int)

    # Approximate inverse prevalence weighting for the positive class.
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())

    if n_pos == 0:
        raise ValueError("Training split contains no fraud examples.")

    scale_pos_weight = n_neg / n_pos
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    # Validation is supplied so we can monitor out-of-sample performance.
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_score = model.predict_proba(X_val)[:, 1]

    print("\n=== Threshold comparison (validation) ===")

    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
      pred = (val_score >= threshold).astype(int)

      precision = precision_score(y_val, pred, zero_division=0)
      recall = recall_score(y_val, pred, zero_division=0)
      f1 = f1_score(y_val, pred, zero_division=0)

      print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F1={f1:.3f}"
     )

    print("\n=== Cost comparison (validation) ===")

    FN_COST = 10
    FP_COST = 1

    best_threshold = None
    best_cost = float("inf")

    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
      pred = (val_score >= threshold).astype(int)

      tn, fp, fn, tp = confusion_matrix(
          y_val, pred, labels=[0, 1]
      ).ravel()

      total_cost = (fn * FN_COST) + (fp * FP_COST)

      print(
        f"Threshold={threshold:.2f} | "
        f"FP={fp} | FN={fn} | "
        f"Cost={total_cost}"
      )

      if total_cost < best_cost:
          best_cost = total_cost
          best_threshold = threshold

    print(
       f"\nSelected threshold: {best_threshold:.2f} "
       f"(validation cost={best_cost})"
    )
    
    test_score = model.predict_proba(X_test)[:, 1]

    val_pred = (val_score >= best_threshold).astype(int)
    test_pred = (test_score >= best_threshold).astype(int)
    
    print("\n=== Test error analysis by fraud type ===")

    test_analysis = test.copy()
    test_analysis["prediction"] = test_pred

    fraud_test = test_analysis[test_analysis["is_fraud"] == 1]

    for fraud_type, group in fraud_test.groupby("fraud_type"):
       total = len(group)
       caught = int(group["prediction"].sum())
       missed = total - caught

       print(
         f"{fraud_type}: "
         f"total={total} | "
         f"caught={caught} | "
         f"missed={missed} | "
         f"recall={caught / total:.3f}"
       )

    evaluate(
         y_val,
         val_pred,
         val_score,
         f"XGBoost - validation (threshold={best_threshold})",
)

    evaluate(
         y_test,
         test_pred,
         test_score,
         f"XGBoost - test (threshold={best_threshold})",
    )

    print("\n=== Feature importances ===")
    importances = (
        pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
    )
    print(importances.to_string(index=False))


if __name__ == "__main__":
    main()


