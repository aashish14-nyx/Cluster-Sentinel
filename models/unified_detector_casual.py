"""Cluster Sentinel - unified ML + graph investigation detector.

The XGBoost model is the primary transaction-level detector.
The graph engine is a separate causal investigation/detection layer
that scores transactions chronologically using only prior state.

Important evaluation rule:
- XGBoost threshold is selected only on validation data.
- The graph engine is label-blind; labels are used only for evaluation.
- The graph layer is strictly time-aware/causal for transaction scoring.
"""

import argparse
import sys
from pathlib import Path

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

# Load the strictly causal rich graph engine from the project's graph folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = PROJECT_ROOT / "graph"
if str(GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_DIR))

from ring_detection_rich_causal import process_stream  # noqa: E402

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

THRESHOLDS = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90,
]

FN_COST = 10
FP_COST = 1


def time_based_split(df, train_frac=0.70, val_frac=0.15):
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1.")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def validate_input(df):
    required = FEATURE_COLS + [
        "transaction_id",
        "customer_id",
        "device_id",
        "instrument_id",
        "timestamp",
        "is_fraud",
        "fraud_type",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not df["transaction_id"].is_unique:
        raise ValueError("Duplicate transaction_id values detected.")
    if df[FEATURE_COLS + ["is_fraud"]].isna().any().any():
        raise ValueError("Missing values detected in model inputs.")
    if not df["is_fraud"].isin([0, 1]).all():
        raise ValueError("is_fraud must contain only 0/1 values.")


def select_threshold_by_cost(y_val, val_score):
    best_threshold = None
    best_cost = float("inf")

    print("\n=== Threshold / cost selection (validation only) ===")
    for threshold in THRESHOLDS:
        pred = (val_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_val, pred, labels=[0, 1]
        ).ravel()
        cost = (fn * FN_COST) + (fp * FP_COST)
        precision = precision_score(y_val, pred, zero_division=0)
        recall = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred, zero_division=0)
        print(
            f"Threshold={threshold:.2f} | Precision={precision:.3f} "
            f"| Recall={recall:.3f} | F1={f1:.3f} "
            f"| FP={fp} | FN={fn} | Cost={cost}"
        )
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    print(
        f"\nSelected threshold: {best_threshold:.2f} "
        f"(validation cost={best_cost}, FN_COST={FN_COST}, FP_COST={FP_COST})"
    )
    return best_threshold


def evaluate_binary(y_true, pred, score, label):
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    pr_auc = average_precision_score(y_true, score)
    roc_auc = roc_auc_score(y_true, score)
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
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


def evaluate_flag(y_true, pred, label):
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    print(f"\n=== {label} ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def typology_report(test, prediction_col, label):
    print(f"\n=== {label}: fraud-type coverage ===")
    fraud = test[test["is_fraud"] == 1]
    for fraud_type, group in fraud.groupby("fraud_type"):
        total = len(group)
        caught = int(group[prediction_col].sum())
        print(
            f"{fraud_type}: total={total} | caught={caught} | "
            f"missed={total - caught} | recall={caught / total:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="../features/features.csv")
    parser.add_argument(
        "--txn-out",
        default="../features/unified_detector_test.csv",
        help="Unified transaction-level output.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    validate_input(df)
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)

    train, val, test = time_based_split(df)
    history = pd.concat([train, val], ignore_index=True)

    print(
        f"Train: {len(train):,} | Validation: {len(val):,} | "
        f"Test: {len(test):,}"
    )
    print("Using canonical causal transaction features for XGBoost.")

    # ------------------------------
    # 1. XGBoost transaction detector
    # ------------------------------
    X_train = train[FEATURE_COLS]
    y_train = train["is_fraud"].astype(int)
    X_val = val[FEATURE_COLS]
    y_val = val["is_fraud"].astype(int)
    X_test = test[FEATURE_COLS]
    y_test = test["is_fraud"].astype(int)

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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    threshold = select_threshold_by_cost(y_val, val_score)

    test_xgb_flag = (test_score >= threshold).astype(int)
    evaluate_binary(
        y_test,
        test_xgb_flag,
        test_score,
        f"XGBoost test (threshold={threshold:.2f})",
    )

    # ------------------------------
    # 2. Strictly causal rich graph layer
    # ------------------------------
    print("\nBuilding CAUSAL rich graph stream (prior state only)...")
    print("IP is temporal corroboration only; it is NOT a persistent graph edge.")

    # Warm graph state using only train + validation history.
    _, graph, history_state = process_stream(
        history,
        min_instrument_customers=2,
        recent_window_seconds=30 * 60,
    )

    # Score test chronologically. The engine scores each row before adding
    # that row to state, so future test rows cannot influence earlier rows.
    graph_test, _, _ = process_stream(
        test,
        min_instrument_customers=2,
        recent_window_seconds=30 * 60,
        initial_state=history_state,
    )

    graph_flag = graph_test["graph_prediction"].astype(int).to_numpy()

    evaluate_flag(y_test, graph_flag, "Graph ring detector - test (CAUSAL)")
    typology_report(graph_test, "graph_prediction", "Graph ring detector")

    ring_mask = graph_test["fraud_type"].eq("shared_instrument_ring")
    ring_total = int(ring_mask.sum())
    ring_caught = int(graph_test.loc[ring_mask, "graph_prediction"].sum())
    print(
        f"\nCausal ring recall: {ring_caught}/{ring_total} = "
        f"{(ring_caught / ring_total if ring_total else 0):.3f}"
    )

    print(
        f"Historical graph after warm-up: "
        f"{graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges"
    )

    # ------------------------------
    # 3. Unified investigation layer
    # ------------------------------
    out = graph_test.copy()
    out["graph_ring_flag"] = graph_flag
    out["ml_score"] = test_score
    out["ml_flag"] = test_xgb_flag

    # The graph signal is a separate investigation signal.
    out["combined_flag"] = (
        out["ml_flag"].astype(int) | out["graph_ring_flag"].astype(int)
    )

    def classify(row):
        ml = int(row["ml_flag"])
        graph = int(row["graph_ring_flag"])
        if ml and graph:
            return "HIGH_RISK_ML_AND_NETWORK"
        if ml:
            return "HIGH_RISK_ML"
        if graph:
            return "NETWORK_RING_RISK"
        return "LOW_RISK"

    out["risk_category"] = out.apply(classify, axis=1)
    out["investigation_reason"] = ""

    out.loc[out["ml_flag"].eq(1), "investigation_reason"] = (
        "high_ml_fraud_score"
    )
    both = out["ml_flag"].eq(1) & out["graph_ring_flag"].eq(1)
    graph_only = out["ml_flag"].eq(0) & out["graph_ring_flag"].eq(1)
    out.loc[both, "investigation_reason"] = "high_ml_fraud_score + suspicious_ring"
    out.loc[graph_only, "investigation_reason"] = "suspicious_ring"

    combined_metrics = evaluate_flag(
        y_test,
        out["combined_flag"].astype(int),
        "Combined ML OR graph detector - test",
    )
    typology_report(out, "combined_flag", "Combined detector")

    print("\n=== Incremental graph contribution over XGBoost ===")
    ml_only = int((out["ml_flag"] == 1).sum())
    graph_only = int(
        ((out["ml_flag"] == 0) & (out["graph_ring_flag"] == 1)).sum()
    )
    both_count = int(
        ((out["ml_flag"] == 1) & (out["graph_ring_flag"] == 1)).sum()
    )
    print(f"XGBoost flagged:       {ml_only:,}")
    print(f"Graph-only flags:      {graph_only:,}")
    print(f"Flags from both:       {both_count:,}")
    print(
        "Interpretation: graph-only flags are the transactions where the graph "
        "layer adds detection beyond the ML threshold."
    )

    # Save unified transaction-level output.
    txn_path = Path(args.txn_out)
    txn_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(txn_path, index=False)

    print(f"\nWrote unified transactions: {txn_path}")

    # Final compact summary suitable for README notes.
    print("\n=== Final test summary ===")
    print(
        f"XGBoost:  P={precision_score(y_test, test_xgb_flag, zero_division=0):.3f} "
        f"R={recall_score(y_test, test_xgb_flag, zero_division=0):.3f} "
        f"F1={f1_score(y_test, test_xgb_flag, zero_division=0):.3f}"
    )
    print(
        f"Graph:    P={precision_score(y_test, graph_flag, zero_division=0):.3f} "
        f"R={recall_score(y_test, graph_flag, zero_division=0):.3f} "
        f"F1={f1_score(y_test, graph_flag, zero_division=0):.3f}"
    )
    print(
        f"Combined: P={combined_metrics['precision']:.3f} "
        f"R={combined_metrics['recall']:.3f} "
        f"F1={combined_metrics['f1']:.3f}"
    )


if __name__ == "__main__":
    main()