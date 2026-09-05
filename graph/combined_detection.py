"""Cluster Sentinel - combined XGBoost + graph detector.

XGBoost provides the primary transaction-level fraud score.
The time-aware graph detector adds a causal shared-instrument signal.

Threshold selection is performed ONLY on validation data.
The selected threshold is then frozen and applied to the untouched test set.
"""

import argparse
from collections import defaultdict, deque

import networkx as nx
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
        "ip_bucket",
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


def evaluate(y_true, y_pred, y_score, label):
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


def select_threshold_by_cost(y_val, val_score):
    print("\n=== Cost comparison (validation) ===")

    best_threshold = None
    best_cost = float("inf")

    for threshold in THRESHOLDS:
        pred = (val_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_val, pred, labels=[0, 1]
        ).ravel()

        total_cost = (fn * FN_COST) + (fp * FP_COST)

        print(
            f"Threshold={threshold:.2f} | "
            f"FP={fp} | FN={fn} | Cost={total_cost}"
        )

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold

    print(
        f"\nSelected threshold: {best_threshold:.2f} "
        f"(validation cost={best_cost})"
    )

    return best_threshold


def build_time_aware_graph_state(
    df,
    min_instrument_customers=2,
    recent_window_seconds=30 * 60,
    initial_state=None,
):
    """
    Score transactions in chronological order using only prior state.
    Returns graph predictions and the final state.
    """
    graph = nx.Graph()
    predictions = []

    if initial_state is None:
        instrument_customers = defaultdict(set)
        ip_customers = defaultdict(set)
        instrument_recent = defaultdict(deque)
    else:
        instrument_customers = defaultdict(
            set,
            {
                k: set(v)
                for k, v in initial_state["instrument_customers"].items()
            },
        )
        ip_customers = defaultdict(
            set,
            {
                k: set(v)
                for k, v in initial_state["ip_customers"].items()
            },
        )
        instrument_recent = defaultdict(
            deque,
            {
                k: deque(v)
                for k, v in initial_state["instrument_recent"].items()
            },
        )

    for row in df.itertuples(index=False):
        ts = int(row.timestamp)
        instrument = row.instrument_id
        customer = row.customer_id
        ip = row.ip_bucket

        prior_customers = instrument_customers[instrument]
        prior_ip_customers = ip_customers[ip]
        recent = instrument_recent[instrument]

        while recent and recent[0] < ts - recent_window_seconds:
            recent.popleft()

        shared_instrument = len(prior_customers) >= min_instrument_customers
        shared_ip = (
            customer not in prior_ip_customers
            and len(prior_ip_customers) >= 1
        )
        recent_burst = len(recent) >= 1

        flagged = shared_instrument and (shared_ip or recent_burst)
        predictions.append(int(flagged))

        # Update only after scoring the current transaction.
        prior_customers.add(customer)
        prior_ip_customers.add(customer)
        recent.append(ts)

        customer_node = f"customer:{customer}"
        instrument_node = f"instrument:{instrument}"
        device_node = f"device:{row.device_id}"
        ip_node = f"ip:{ip}"

        graph.add_node(customer_node, node_type="customer")
        graph.add_node(instrument_node, node_type="instrument")
        graph.add_node(device_node, node_type="device")
        graph.add_node(ip_node, node_type="ip")

        graph.add_edge(customer_node, instrument_node, timestamp=ts)
        graph.add_edge(customer_node, device_node, timestamp=ts)
        graph.add_edge(customer_node, ip_node, timestamp=ts)

    state = {
        "instrument_customers": dict(instrument_customers),
        "ip_customers": dict(ip_customers),
        "instrument_recent": {
            k: list(v) for k, v in instrument_recent.items()
        },
    }

    return predictions, graph, state


def report_typology_results(test_df):
    print("\n=== Combined test error analysis by fraud type ===")

    fraud_test = test_df[test_df["is_fraud"] == 1]

    for fraud_type, group in fraud_test.groupby("fraud_type"):
        total = len(group)
        caught = int(group["combined_prediction"].sum())
        missed = total - caught

        print(
            f"{fraud_type}: "
            f"total={total} | "
            f"caught={caught} | "
            f"missed={missed} | "
            f"recall={caught / total:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=str,
        default="../features/features.csv",
    )
    parser.add_argument(
        "--min-instrument-customers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--recent-window-minutes",
        type=int,
        default=30,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    validate_input(df)

    train, val, test = time_based_split(df)

    print(
        f"Train: {len(train):,} rows "
        f"({int(train['is_fraud'].sum())} fraud, {train['is_fraud'].mean():.3%})"
    )
    print(
        f"Validation: {len(val):,} rows "
        f"({int(val['is_fraud'].sum())} fraud, {val['is_fraud'].mean():.3%})"
    )
    print(
        f"Test: {len(test):,} rows "
        f"({int(test['is_fraud'].sum())} fraud, {test['is_fraud'].mean():.3%})"
    )

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

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    best_threshold = select_threshold_by_cost(y_val, val_score)

    # Build the causal graph from training + validation history.
    history = pd.concat([train, val], ignore_index=True)

    _, graph, history_state = build_time_aware_graph_state(
        history,
        min_instrument_customers=args.min_instrument_customers,
        recent_window_seconds=args.recent_window_minutes * 60,
    )

    # Score the test stream chronologically using ONLY historical state.
    graph_predictions, _, _ = build_time_aware_graph_state(
        test,
        min_instrument_customers=args.min_instrument_customers,
        recent_window_seconds=args.recent_window_minutes * 60,
        initial_state=history_state,
    )

    graph_predictions = pd.Series(
        graph_predictions,
        index=test.index,
        dtype="int64",
    )

    xgb_test_prediction = (test_score >= best_threshold).astype(int)

    combined_prediction = (
        (xgb_test_prediction == 1)
        | (graph_predictions.to_numpy() == 1)
    ).astype(int)

    test_result = test.copy()
    test_result["xgb_prediction"] = xgb_test_prediction
    test_result["graph_prediction"] = graph_predictions.to_numpy()
    test_result["combined_prediction"] = combined_prediction

    evaluate(
        y_test,
        xgb_test_prediction,
        test_score,
        f"XGBoost only - test (threshold={best_threshold})",
    )

    # The graph is a binary signal, so use it for decision analysis rather
    # than treating it as a calibrated probability score.
    graph_tp = int(
        ((y_test == 1) & (graph_predictions.to_numpy() == 1)).sum()
    )
    graph_fp = int(
        ((y_test == 0) & (graph_predictions.to_numpy() == 1)).sum()
    )
    graph_fn = int(
        ((y_test == 1) & (graph_predictions.to_numpy() == 0)).sum()
    )

    graph_precision = (
        graph_tp / (graph_tp + graph_fp)
        if graph_tp + graph_fp
        else 0.0
    )
    graph_recall = (
        graph_tp / (graph_tp + graph_fn)
        if graph_tp + graph_fn
        else 0.0
    )
    graph_f1 = (
        2 * graph_precision * graph_recall
        / (graph_precision + graph_recall)
        if graph_precision + graph_recall
        else 0.0
    )

    print("\n=== Graph signal - test ===")
    print(f"TP={graph_tp} FP={graph_fp} FN={graph_fn}")
    print(f"Precision: {graph_precision:.3f}")
    print(f"Recall:    {graph_recall:.3f}")
    print(f"F1:        {graph_f1:.3f}")

    evaluate(
        y_test,
        combined_prediction,
        test_score,
        "Combined XGBoost + graph - test",
    )

    print(
        f"\nGraph historical warm-up: "
        f"{graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges"
    )

    report_typology_results(test_result)

    ring = test_result[
        test_result["fraud_type"] == "shared_instrument_ring"
    ]

    if not ring.empty:
        xgb_ring_recall = ring["xgb_prediction"].mean()
        graph_ring_recall = ring["graph_prediction"].mean()
        combined_ring_recall = ring["combined_prediction"].mean()

        print("\n=== Ring recall comparison ===")
        print(f"XGBoost only: {xgb_ring_recall:.3f}")
        print(f"Graph only:   {graph_ring_recall:.3f}")
        print(f"Combined:     {combined_ring_recall:.3f}")


if __name__ == "__main__":
    main()
