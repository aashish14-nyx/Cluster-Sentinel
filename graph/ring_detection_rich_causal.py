"""Cluster Sentinel - causal rich graph-based ring detector.

Processes transactions chronologically. Each transaction is scored using only
state from earlier transactions, then the current transaction updates state.
IP reuse is a temporal corroborating signal, never a persistent graph edge.
"""

import argparse
from collections import defaultdict, deque

import networkx as nx
import pandas as pd


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
    required = [
        "transaction_id", "customer_id", "device_id", "ip_bucket",
        "instrument_id", "timestamp", "is_fraud", "fraud_type"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not df["transaction_id"].is_unique:
        raise ValueError("Duplicate transaction_id values detected.")


def process_stream(
    df,
    min_instrument_customers=2,
    recent_window_seconds=30 * 60,
    initial_state=None,
):
    """Score each transaction using only prior graph/stream state."""
    graph = nx.Graph()
    predictions = []

    if initial_state is None:
        instrument_customers = defaultdict(set)
        ip_customers = defaultdict(set)
        instrument_recent = defaultdict(deque)
    else:
        instrument_customers = defaultdict(
            set,
            {k: set(v) for k, v in initial_state["instrument_customers"].items()},
        )
        ip_customers = defaultdict(
            set,
            {k: set(v) for k, v in initial_state["ip_customers"].items()},
        )
        instrument_recent = defaultdict(
            deque,
            {k: deque(v) for k, v in initial_state["instrument_recent"].items()},
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
        shared_ip = customer not in prior_ip_customers and len(prior_ip_customers) >= 1
        recent_burst = len(recent) >= 1

        flagged = shared_instrument and (shared_ip or recent_burst)
        predictions.append(int(flagged))

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
        # IP is temporal corroboration only; it is not a persistent graph edge.

    state = {
        "instrument_customers": dict(instrument_customers),
        "ip_customers": dict(ip_customers),
        "instrument_recent": {k: list(v) for k, v in instrument_recent.items()},
    }

    result = df.copy()
    result["graph_prediction"] = predictions
    return result, graph, state


def report_results(df):
    flagged = int(df["graph_prediction"].sum())
    fraud = int(df["is_fraud"].sum())
    tp = int(((df["is_fraud"] == 1) & (df["graph_prediction"] == 1)).sum())
    fp = int(((df["is_fraud"] == 0) & (df["graph_prediction"] == 1)).sum())
    fn = int(((df["is_fraud"] == 1) & (df["graph_prediction"] == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / fraud if fraud else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print("\n=== Overall graph detector ===")
    print(f"Flagged transactions: {flagged}")
    print(f"TP={tp} FP={fp} FN={fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")


def report_ring_results(df):
    ring = df[
        (df["is_fraud"] == 1)
        & (df["fraud_type"] == "shared_instrument_ring")
    ]
    if ring.empty:
        print("\nNo shared_instrument_ring rows found.")
        return

    caught = int(ring["graph_prediction"].sum())
    total = len(ring)

    print("\n=== Time-aware graph ring detection ===")
    print(f"Ring fraud transactions: {total}")
    print(f"Caught: {caught}")
    print(f"Missed: {total - caught}")
    print(f"Recall: {caught / total:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="../features/features.csv")
    parser.add_argument("--min-instrument-customers", type=int, default=2)
    parser.add_argument("--recent-window-minutes", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    validate_input(df)

    train, val, test = time_based_split(df)

    print(
        f"Train: {len(train):,} rows | "
        f"Validation: {len(val):,} rows | "
        f"Test: {len(test):,} rows"
    )

    history = pd.concat([train, val], ignore_index=True)

    _, graph, history_state = process_stream(
        history,
        min_instrument_customers=args.min_instrument_customers,
        recent_window_seconds=args.recent_window_minutes * 60,
    )

    test_result, _, _ = process_stream(
        test,
        min_instrument_customers=args.min_instrument_customers,
        recent_window_seconds=args.recent_window_minutes * 60,
        initial_state=history_state,
    )

    print(
        f"\nGraph after historical warm-up: "
        f"{graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges"
    )

    report_results(test_result)
    report_ring_results(test_result)

    print("\n=== Ring fraud by test outcome ===")
    ring = test_result[test_result["fraud_type"] == "shared_instrument_ring"]
    print(f"caught: {int((ring['graph_prediction'] == 1).sum())}")
    print(f"missed: {int((ring['graph_prediction'] == 0).sum())}")


if __name__ == "__main__":
    main()