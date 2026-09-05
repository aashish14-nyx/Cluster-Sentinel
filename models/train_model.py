"""Cluster Sentinel - time-based Logistic Regression and XGBoost baseline.

Uses the corrected causal features from build_features_v2.py.
Split is chronological: train / validation / test.
Validation is reserved for model/threshold selection; test is untouched until final evaluation.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

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
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)

    if not np.isclose(train_frac + val_frac, 0.85):
        raise ValueError("This script expects train_frac + val_frac = 0.85.")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def evaluate(y_true, y_pred, y_score, label):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

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
    missing = [c for c in FEATURE_COLS + ["timestamp", "is_fraud"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[FEATURE_COLS + ["is_fraud"]].isna().any().any():
        raise ValueError("Missing values detected in model inputs.")

    if not df["transaction_id"].is_unique:
        raise ValueError("Duplicate transaction_id values detected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="../features/features_v2.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    validate_input(df)

    train, val, test = time_based_split(df)

    for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
        print(
            f"{name}: {len(part):,} rows | "
            f"{int(part['is_fraud'].sum()):,} fraud | "
            f"{part['is_fraud'].mean():.3%} fraud"
        )

    X_train, y_train = train[FEATURE_COLS], train["is_fraud"].astype(int)
    X_val, y_val = val[FEATURE_COLS], val["is_fraud"].astype(int)
    X_test, y_test = test[FEATURE_COLS], test["is_fraud"].astype(int)

    # Logistic Regression needs scaled numeric features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    val_score = model.predict_proba(X_val_scaled)[:, 1]
    test_score = model.predict_proba(X_test_scaled)[:, 1]

    # 0.5 is only a reference point. Validation is the place to tune later.
    val_pred = (val_score >= args.threshold).astype(int)
    test_pred = (test_score >= args.threshold).astype(int)

    evaluate(
        y_val,
        val_pred,
        val_score,
        f"Logistic Regression - validation (threshold={args.threshold})",
    )
    evaluate(
        y_test,
        test_pred,
        test_score,
        f"Logistic Regression - test (threshold={args.threshold})",
    )

    print("\n=== Standardized Logistic Regression coefficients ===")
    coef_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "coefficient": model.coef_[0],
    })
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    print(
        coef_df.sort_values("abs_coefficient", ascending=False)
        .drop(columns="abs_coefficient")
        .to_string(index=False)
    )

    # XGBoost is intentionally optional so the script can still run without it.
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("\nXGBoost not installed. Install with: pip install xgboost")
        return

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    xgb_val_score = xgb.predict_proba(X_val)[:, 1]
    xgb_test_score = xgb.predict_proba(X_test)[:, 1]

    xgb_val_pred = (xgb_val_score >= args.threshold).astype(int)
    xgb_test_pred = (xgb_test_score >= args.threshold).astype(int)

    evaluate(
        y_val,
        xgb_val_pred,
        xgb_val_score,
        f"XGBoost - validation (threshold={args.threshold})",
    )
    evaluate(
        y_test,
        xgb_test_pred,
        xgb_test_score,
        f"XGBoost - test (threshold={args.threshold})",
    )

    print("\n=== XGBoost feature importance ===")
    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": xgb.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
