"""Cluster Sentinel - baseline hand-written fraud rules.

Uses the corrected causal feature names from build_features_v2.py.

Rules:
  R1 - velocity: velocity_1h >= 3
  R2 - takeover: new device + new instrument + unusually large amount
  R3 - ring: instrument_customer_count_before >= 3

A transaction is flagged if ANY rule fires.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def apply_baseline_rules(df, takeover_amount_ratio=np.e):
    """Apply simple rules using only causal/model-ready features."""
    r1_velocity = df["velocity_1h"] >= 3

    # amount_log_ratio = log(amount / customer historical average)
    # > log(e) = 1 means amount is > e (~2.72x) the customer's baseline.
    r2_takeover = (
        (df["is_new_device"] == 1)
        & (df["is_new_instrument"] == 1)
        & (df["amount_log_ratio"] > np.log(takeover_amount_ratio))
    )

    # Causal count: only customers seen before the current transaction.
    r3_ring = df["instrument_customer_count_before"] >= 3

    flagged = r1_velocity | r2_takeover | r3_ring

    rule_hits = pd.DataFrame({
        "velocity": r1_velocity,
        "takeover": r2_takeover,
        "ring": r3_ring,
    }, index=df.index)

    return flagged, rule_hits


def evaluate(y_true, y_pred, label="baseline rules"):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="../features/features_v2.csv")
    parser.add_argument(
        "--takeover-ratio",
        type=float,
        default=np.e,
        help="Amount/customer-baseline ratio threshold for takeover rule (default: e ≈ 2.72x).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.features)

    required = [
        "velocity_1h",
        "is_new_device",
        "is_new_instrument",
        "amount_log_ratio",
        "instrument_customer_count_before",
        "is_fraud",
        "fraud_type",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    flagged, rule_hits = apply_baseline_rules(
        df, takeover_amount_ratio=args.takeover_ratio
    )

    evaluate(
        df["is_fraud"],
        flagged,
        label="Baseline rules (all typologies combined)",
    )

    # Show each rule independently on the full dataset.
    print("\n=== Individual rule performance ===")
    for rule_name in ["velocity", "takeover", "ring"]:
        evaluate(
            df["is_fraud"],
            rule_hits[rule_name],
            label=f"Rule: {rule_name}",
        )

    # Among true fraud, show which rules fired. A transaction can match
    # multiple rules, so we report all hits rather than forcing one winner.
    print("\n=== Rule hits among true fraud ===")
    fraud_mask = df["is_fraud"].astype(bool)

    for fraud_type in sorted(df.loc[fraud_mask, "fraud_type"].unique()):
        mask = fraud_mask & (df["fraud_type"] == fraud_type)
        print(f"\n{fraud_type}: {int(mask.sum())} fraud rows")
        for rule_name in ["velocity", "takeover", "ring"]:
            hits = int(rule_hits.loc[mask, rule_name].sum())
            print(f"  {rule_name}: {hits} ({hits / mask.sum():.1%})")

    # False positives.
    fp_mask = flagged & ~fraud_mask
    print("\n=== False positives ===")
    print(
        f"Total false positives: {int(fp_mask.sum())} "
        f"({fp_mask.mean():.3%} of all transactions)"
    )


if __name__ == "__main__":
    main()