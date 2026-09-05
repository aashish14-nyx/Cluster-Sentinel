"""Cluster Sentinel - causal graph ring investigation.

This is an INVESTIGATION layer, not a detector.

For each graph-flagged test transaction, reconstruct the graph state that
existed BEFORE that transaction and report the historical evidence that
caused the causal graph to flag it.

Mirrors the causal rich graph logic:
    shared_instrument = prior distinct customers on instrument >= 2
    shared_ip        = current customer is new to the IP and the IP has >= 1 prior customer
    recent_burst     = >= 1 prior transaction on the instrument within 30 minutes
    graph_flag       = shared_instrument AND (shared_ip OR recent_burst)

Fraud labels are evaluation-only and never form graph evidence.
"""

import argparse
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

MIN_INSTRUMENT_CUSTOMERS = 2
RECENT_WINDOW_SECONDS = 30 * 60

REQUIRED_DETECTOR = [
    "transaction_id", "customer_id", "device_id", "instrument_id",
    "timestamp", "graph_ring_flag"
]

REQUIRED_HISTORY = [
    "transaction_id", "customer_id", "device_id", "instrument_id",
    "ip_bucket", "timestamp"
]


def validate(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

    if not df["transaction_id"].is_unique:
        raise ValueError(f"{name} contains duplicate transaction_id values.")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise").astype("int64")


def reconstruct_history(history_df):
    """Return the pre-transaction causal state for every transaction."""
    instrument_customers = defaultdict(set)
    ip_customers = defaultdict(set)
    instrument_recent = defaultdict(deque)
    instrument_devices = defaultdict(set)

    rows = (
        history_df
        .sort_values(["timestamp", "transaction_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    evidence = []

    for row in rows.itertuples(index=False):
        ts = int(row.timestamp)
        txn = str(row.transaction_id)
        customer = str(row.customer_id)
        instrument = str(row.instrument_id)
        device = str(row.device_id)
        ip = str(row.ip_bucket)

        prior_customers = instrument_customers[instrument]
        prior_ip_customers = ip_customers[ip]
        recent = instrument_recent[instrument]

        # Expire transactions outside the same 30-minute causal window.
        while recent and recent[0][0] < ts - RECENT_WINDOW_SECONDS:
            recent.popleft()

        # ---------------- PRE-TRANSACTION STATE ----------------
        prior_recent = list(recent)

        shared_instrument = len(prior_customers) >= MIN_INSTRUMENT_CUSTOMERS
        shared_ip = (
            customer not in prior_ip_customers
            and len(prior_ip_customers) >= 1
        )
        recent_burst = len(prior_recent) >= 1

        reconstructed_flag = int(
            shared_instrument and (shared_ip or recent_burst)
        )

        evidence.append({
            "transaction_id": txn,
            "timestamp": ts,
            "instrument_id": instrument,
            "customer_id": customer,
            "device_id": device,
            "prior_instrument_customer_count": len(prior_customers),
            "prior_instrument_customers": "|".join(sorted(prior_customers)),
            "prior_instrument_device_count": len(instrument_devices[instrument]),
            "prior_instrument_devices": "|".join(
                sorted(instrument_devices[instrument])
            ),
            "prior_transactions_within_30m": len(prior_recent),
            "prior_transaction_ids_within_30m": "|".join(
                item[1] for item in prior_recent
            ),
            "prior_transaction_customers_within_30m": "|".join(
                item[2] for item in prior_recent
            ),
            "prior_ip_customer_count": len(prior_ip_customers),
            "prior_ip_customers": "|".join(sorted(prior_ip_customers)),
            "shared_instrument_condition": int(shared_instrument),
            "shared_ip_condition": int(shared_ip),
            "recent_burst_condition": int(recent_burst),
            "reconstructed_graph_flag": reconstructed_flag,
        })

        # ---------------- POST-SCORING UPDATE ----------------
        prior_customers.add(customer)
        instrument_devices[instrument].add(device)
        ip_customers[ip].add(customer)
        recent.append((ts, txn, customer))

    return pd.DataFrame(evidence)


def make_trigger_report(detector_df, evidence_df):
    flagged = detector_df[detector_df["graph_ring_flag"].eq(1)].copy()

    if flagged.empty:
        return flagged

    report = flagged.merge(
        evidence_df,
        on=[
            "transaction_id", "timestamp", "instrument_id",
            "customer_id", "device_id"
        ],
        how="left",
        validate="one_to_one",
        suffixes=("_test", "_history"),
    )

    if report["reconstructed_graph_flag"].isna().any():
        raise RuntimeError("Some graph-flagged transactions have no history evidence.")

    # Critical safety check: the reconstructed causal state must reproduce
    # every detector graph flag.
    mismatches = report[
        report["graph_ring_flag"].astype(int)
        != report["reconstructed_graph_flag"].astype(int)
    ]

    if not mismatches.empty:
        cols = [
            "transaction_id", "graph_ring_flag", "reconstructed_graph_flag",
            "prior_instrument_customer_count", "prior_transactions_within_30m",
            "shared_instrument_condition", "shared_ip_condition",
            "recent_burst_condition"
        ]
        raise RuntimeError(
            "Causal reconstruction mismatch:\n"
            + mismatches[cols].to_string(index=False)
        )

    def reason(row):
        reasons = []

        if int(row["shared_instrument_condition"]):
            reasons.append(
                f"the instrument already had "
                f"{int(row['prior_instrument_customer_count'])} prior distinct customer(s)"
            )

        if int(row["recent_burst_condition"]):
            reasons.append(
                f"{int(row['prior_transactions_within_30m'])} prior "
                "transaction(s) on the instrument were within 30 minutes"
            )

        if int(row["shared_ip_condition"]):
            reasons.append(
                f"the IP bucket had {int(row['prior_ip_customer_count'])} "
                "prior customer(s)"
            )

        return "; ".join(reasons) + "."

    report["investigation_reason"] = report.apply(reason, axis=1)

    return report.sort_values(
        ["timestamp", "transaction_id"], kind="mergesort"
    ).reset_index(drop=True)


def make_cluster_summary(report, window_minutes):
    """Cluster only the graph-triggered transactions for investigator view."""
    if report.empty:
        return pd.DataFrame()

    work = report.sort_values(
        ["instrument_id", "timestamp", "transaction_id"],
        kind="mergesort"
    ).copy()

    work["previous_trigger_ts"] = (
        work.groupby("instrument_id")["timestamp"].shift(1)
    )

    work["new_cluster"] = (
        work["previous_trigger_ts"].isna()
        | (
            work["timestamp"] - work["previous_trigger_ts"]
            > window_minutes * 60
        )
    )

    work["cluster_seq"] = (
        work.groupby("instrument_id")["new_cluster"]
        .cumsum()
        .astype(int)
    )

    work["cluster_id"] = (
        work["instrument_id"].astype(str)
        + "_cluster_"
        + work["cluster_seq"].astype(str)
    )

    out = []

    for cluster_id, group in work.groupby("cluster_id", sort=False):
        out.append({
            "cluster_id": cluster_id,
            "instrument_id": str(group["instrument_id"].iloc[0]),
            "trigger_count": len(group),
            "trigger_transaction_ids": "|".join(
                group["transaction_id"].astype(str)
            ),
            "trigger_customers": "|".join(
                sorted(group["customer_id"].astype(str).unique())
            ),
            "max_prior_instrument_customers": int(
                group["prior_instrument_customer_count"].max()
            ),
            "max_prior_transactions_within_30m": int(
                group["prior_transactions_within_30m"].max()
            ),
            "any_recent_burst": int(group["recent_burst_condition"].max()),
            "any_shared_ip": int(group["shared_ip_condition"].max()),
            "fraud_count_evaluation_only": (
                int(group["is_fraud"].sum())
                if "is_fraud" in group.columns else None
            ),
            "ring_fraud_count_evaluation_only": (
                int(
                    (
                        group["fraud_type"]
                        == "shared_instrument_ring"
                    ).sum()
                )
                if "fraud_type" in group.columns else None
            ),
        })

    return pd.DataFrame(out)


def print_report(report):
    print("\n=== CAUSAL GRAPH RING INVESTIGATION REPORT ===")

    if report.empty:
        print("No graph-flagged transactions found.")
        return

    print(f"Graph-triggered transactions: {len(report)}")

    for row in report.itertuples(index=False):
        print("\n" + "=" * 72)
        print(f"Trigger transaction: {row.transaction_id}")
        print(f"Timestamp: {row.timestamp}")
        print(f"Instrument: {row.instrument_id}")
        print(f"Customer: {row.customer_id}")
        print(f"Device: {row.device_id}")

        if "ml_score" in report.columns:
            score = getattr(row, "ml_score", None)
            if score is not None and pd.notna(score):
                print(f"ML score: {float(score):.4f}")

        print("\nHistorical graph evidence AVAILABLE BEFORE trigger:")
        print(
            "  Prior distinct customers on instrument: "
            f"{row.prior_instrument_customer_count}"
        )
        print(
            "  Prior customers: "
            f"{row.prior_instrument_customers or '(none)'}"
        )
        print(
            "  Prior devices on instrument: "
            f"{row.prior_instrument_devices or '(none)'}"
        )
        print(
            "  Prior transactions within 30m: "
            f"{row.prior_transactions_within_30m}"
        )
        print(
            "  Prior transaction IDs: "
            f"{row.prior_transaction_ids_within_30m or '(none)'}"
        )
        print(
            "  Prior transaction customers: "
            f"{row.prior_transaction_customers_within_30m or '(none)'}"
        )
        print(
            "  Prior customers on IP: "
            f"{row.prior_ip_customers or '(none)'}"
        )

        print("\nCausal graph conditions:")
        print(
            "  Shared instrument: "
            f"{int(row.shared_instrument_condition)}"
        )
        print(f"  Shared IP: {int(row.shared_ip_condition)}")
        print(f"  Recent burst: {int(row.recent_burst_condition)}")
        print(
            "  Reconstructed graph flag: "
            f"{int(row.reconstructed_graph_flag)}"
        )

        print("\nInvestigation reason:")
        print(f"  {row.investigation_reason}")

        if "is_fraud" in report.columns:
            value = getattr(row, "is_fraud", None)
            if value is not None and pd.notna(value):
                print(
                    f"\nEvaluation only - fraud label: {int(value)}"
                )

        if "fraud_type" in report.columns:
            value = getattr(row, "fraud_type", None)
            if value is not None and pd.notna(value):
                print(
                    f"Evaluation only - fraud type: {value}"
                )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="../features/unified_detector_test.csv",
    )
    parser.add_argument(
        "--history",
        default="../features/features.csv",
    )
    parser.add_argument(
        "--out",
        default="../features/ring_investigation_test.csv",
    )
    parser.add_argument(
        "--cluster-out",
        default="../features/ring_investigation_test_clusters.csv",
    )
    parser.add_argument(
        "--window-minutes",
        type=float,
        default=30.0,
    )

    args = parser.parse_args()

    detector_df = pd.read_csv(args.input)
    history_df = pd.read_csv(args.history)

    validate(detector_df, REQUIRED_DETECTOR, "Detector input")
    validate(history_df, REQUIRED_HISTORY, "History input")

    flagged_count = int(detector_df["graph_ring_flag"].eq(1).sum())
    print(f"Graph-flagged transactions: {flagged_count}")

    history_df["transaction_id"] = history_df["transaction_id"].astype(str)
    detector_df["transaction_id"] = detector_df["transaction_id"].astype(str)

    # Full chronological history is used only to reconstruct PRE-trigger state.
    evidence_df = reconstruct_history(history_df)

    report = make_trigger_report(detector_df, evidence_df)
    print_report(report)

    cluster_summary = make_cluster_summary(
        report,
        args.window_minutes,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    cluster_out = Path(args.cluster_out)
    cluster_out.parent.mkdir(parents=True, exist_ok=True)
    cluster_summary.to_csv(cluster_out, index=False)

    print(f"\nWrote per-trigger report: {out}")
    print(f"Wrote cluster summary: {cluster_out}")


if __name__ == "__main__":
    main()
