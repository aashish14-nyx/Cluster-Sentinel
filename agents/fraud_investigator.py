"""
Cluster Sentinel - Fraud Investigator

A deterministic, auditable investigator layer that sits AFTER the real-time
ML/graph detector.

Architecture:
    detector -> investigator -> evidence bundle -> investigator decision

The investigator never changes the detector score and never uses fraud labels
to create evidence. Labels are included only as optional evaluation metadata.

Usage from Cluster-Sentinel/agents:
    python fraud_investigator.py \
        --transaction pay_0000309422 \
        --detector ../features/unified_detector_test.csv \
        --history ../features/features.csv \
        --output ../features/investigation_case_pay_0000309422.md

Batch mode:
    python fraud_investigator.py \
        --top-k 10 \
        --detector ../features/unified_detector_test.csv \
        --history ../features/features.csv
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


REQUIRED_DETECTOR = {
    "transaction_id",
    "customer_id",
    "device_id",
    "ip_bucket",
    "instrument_id",
    "amount",
    "timestamp",
    "is_fraud",
    "fraud_type",
    "graph_ring_flag",
    "ml_score",
    "ml_flag",
    "combined_flag",
}

REQUIRED_HISTORY = {
    "transaction_id",
    "customer_id",
    "device_id",
    "ip_bucket",
    "instrument_id",
    "amount",
    "timestamp",
}


@dataclass
class Decision:
    transaction_id: str
    decision: str
    confidence: str
    reasons: list[str]


class FraudInvestigator:
    """Tool-based investigator for post-detection case analysis."""

    def __init__(self, detector_path: str, history_path: str):
        self.detector = pd.read_csv(detector_path)
        self.history = pd.read_csv(history_path)

        missing_detector = REQUIRED_DETECTOR - set(self.detector.columns)
        missing_history = REQUIRED_HISTORY - set(self.history.columns)

        if missing_detector:
            raise ValueError(
                f"Detector file missing columns: {sorted(missing_detector)}"
            )
        if missing_history:
            raise ValueError(
                f"History file missing columns: {sorted(missing_history)}"
            )

        for df in (self.detector, self.history):
            df["transaction_id"] = df["transaction_id"].astype(str)
            df["customer_id"] = df["customer_id"].astype(str)
            df["device_id"] = df["device_id"].astype(str)
            df["ip_bucket"] = df["ip_bucket"].astype(str)
            df["instrument_id"] = df["instrument_id"].astype(str)
            df["timestamp"] = pd.to_numeric(
                df["timestamp"], errors="raise"
            ).astype("int64")

        if not self.detector["transaction_id"].is_unique:
            raise ValueError("Detector contains duplicate transaction IDs.")
        if not self.history["transaction_id"].is_unique:
            raise ValueError("History contains duplicate transaction IDs.")

        self.history = self.history.sort_values(
            ["timestamp", "transaction_id"], kind="mergesort"
        ).reset_index(drop=True)

    def get_transaction(self, transaction_id: str) -> pd.Series:
        rows = self.detector[
            self.detector["transaction_id"].eq(str(transaction_id))
        ]
        if rows.empty:
            raise KeyError(
                f"Transaction {transaction_id} not found in detector output."
            )
        return rows.iloc[0]

    def get_customer_history(self, customer_id: str, before_ts: int) -> pd.DataFrame:
        return self.history[
            self.history["customer_id"].eq(str(customer_id))
            & (self.history["timestamp"] < int(before_ts))
        ].sort_values(["timestamp", "transaction_id"], kind="mergesort")

    def get_device_history(self, device_id: str, before_ts: int) -> pd.DataFrame:
        return self.history[
            self.history["device_id"].eq(str(device_id))
            & (self.history["timestamp"] < int(before_ts))
        ].sort_values(["timestamp", "transaction_id"], kind="mergesort")

    def get_instrument_history(
        self,
        instrument_id: str,
        before_ts: int,
    ) -> pd.DataFrame:
        return self.history[
            self.history["instrument_id"].eq(str(instrument_id))
            & (self.history["timestamp"] < int(before_ts))
        ].sort_values(["timestamp", "transaction_id"], kind="mergesort")

    def get_ip_history(self, ip_bucket: str, before_ts: int) -> pd.DataFrame:
        return self.history[
            self.history["ip_bucket"].eq(str(ip_bucket))
            & (self.history["timestamp"] < int(before_ts))
        ].sort_values(["timestamp", "transaction_id"], kind="mergesort")

    def causal_graph_evidence(
        self,
        instrument_history: pd.DataFrame,
        ip_history: pd.DataFrame,
        current_customer: str,
        trigger_ts: int,
    ) -> dict:
        recent = instrument_history[
            instrument_history["timestamp"] >=
            int(trigger_ts) - 30 * 60
        ]

        prior_customers = sorted(
            instrument_history["customer_id"].unique().tolist()
        )

        prior_devices = sorted(
            instrument_history["device_id"].unique().tolist()
        )

        prior_ip_customers = sorted(
            ip_history["customer_id"].unique().tolist()
        )

        shared_instrument = len(prior_customers) >= 2
        shared_ip = (
            current_customer not in set(prior_ip_customers)
            and len(prior_ip_customers) >= 1
        )
        recent_burst = not recent.empty

        graph_flag = int(
            shared_instrument and (shared_ip or recent_burst)
        )

        return {
            "prior_instrument_customer_count": len(prior_customers),
            "prior_instrument_customers": prior_customers,
            "prior_instrument_device_count": len(prior_devices),
            "prior_instrument_devices": prior_devices,
            "prior_transactions_within_30m": int(len(recent)),
            "prior_transaction_ids_within_30m":
                recent["transaction_id"].astype(str).tolist(),
            "prior_ip_customer_count": len(prior_ip_customers),
            "prior_ip_customers": prior_ip_customers,
            "shared_instrument_condition": int(shared_instrument),
            "shared_ip_condition": int(shared_ip),
            "recent_burst_condition": int(recent_burst),
            "reconstructed_graph_flag": graph_flag,
        }

    def investigate(self, transaction_id: str) -> tuple[dict, Decision]:
        txn = self.get_transaction(transaction_id)

        txn_id = str(txn["transaction_id"])
        customer = str(txn["customer_id"])
        device = str(txn["device_id"])
        instrument = str(txn["instrument_id"])
        ip_bucket = str(txn["ip_bucket"])
        ts = int(txn["timestamp"])

        instrument_history = self.get_instrument_history(instrument, ts)
        customer_history = self.get_customer_history(customer, ts)
        device_history = self.get_device_history(device, ts)
        ip_history = self.get_ip_history(ip_bucket, ts)

        graph = self.causal_graph_evidence(
            instrument_history,
            ip_history,
            customer,
            ts,
        )

        ml_score = float(txn["ml_score"])
        ml_flag = int(txn["ml_flag"])
        graph_flag = int(txn["graph_ring_flag"])
        combined_flag = int(txn["combined_flag"])

        velocity_1h = int(txn["velocity_1h"]) if "velocity_1h" in txn else None
        velocity_24h = int(txn["velocity_24h"]) if "velocity_24h" in txn else None
        time_since_last = (
            float(txn["time_since_last_txn_sec"])
            if "time_since_last_txn_sec" in txn
            and pd.notna(txn["time_since_last_txn_sec"])
            else None
        )
        new_device = int(txn["is_new_device"]) if "is_new_device" in txn else None
        new_instrument = (
            int(txn["is_new_instrument"])
            if "is_new_instrument" in txn
            else None
        )

        reasons = []

        if ml_flag:
            reasons.append(
                f"XGBoost score {ml_score:.4f} is above the production "
                "operating threshold."
            )

        if graph_flag:
            reasons.append(
                "Causal graph evidence was observable before the transaction: "
                f"{graph['prior_instrument_customer_count']} prior customers "
                f"had used the instrument."
            )

        if graph["recent_burst_condition"]:
            reasons.append(
                f"{graph['prior_transactions_within_30m']} prior transaction(s) "
                "were observed on the same instrument within 30 minutes."
            )

        if velocity_1h is not None and velocity_1h >= 3:
            reasons.append(
                f"High short-window velocity: {velocity_1h} prior "
                "transactions in the 1-hour window."
            )

        if new_device and new_instrument:
            reasons.append(
                "Both device and instrument are new to the customer context."
            )

        if len(device_history) >= 2:
            reasons.append(
                f"Device has {len(device_history)} prior transaction(s) "
                "associated with the same customer/device context."
            )

        # Conservative deterministic disposition.
        if ml_flag and graph_flag:
            decision = "ESCALATE_HIGH_RISK"
            confidence = "HIGH"
        elif ml_flag:
            decision = "ESCALATE_ML"
            confidence = "MEDIUM"
        elif graph_flag:
            decision = "ESCALATE_GRAPH"
            confidence = "MEDIUM"
        elif (
            (new_device and new_instrument)
            or (velocity_1h is not None and velocity_1h >= 3)
        ):
            decision = "REVIEW_SUPPORTING_EVIDENCE"
            confidence = "LOW"
        else:
            decision = "NO_ESCALATION"
            confidence = "LOW"

        # Labels are deliberately isolated here as evaluation metadata only.
        evaluation = {
            "is_fraud_evaluation_only": bool(txn["is_fraud"]),
            "fraud_type_evaluation_only": (
                None if pd.isna(txn["fraud_type"])
                else str(txn["fraud_type"])
            ),
        }

        evidence = {
            "transaction": {
                "transaction_id": txn_id,
                "customer_id": customer,
                "device_id": device,
                "instrument_id": instrument,
                "ip_bucket": ip_bucket,
                "amount": float(txn["amount"]),
                "timestamp": ts,
                "ml_score": ml_score,
                "ml_flag": ml_flag,
                "graph_ring_flag": graph_flag,
                "combined_flag": combined_flag,
            },
            "model_signals": {
                "velocity_1h": velocity_1h,
                "velocity_24h": velocity_24h,
                "time_since_last_txn_sec": time_since_last,
                "is_new_device": new_device,
                "is_new_instrument": new_instrument,
                "amount_log_ratio": (
                    float(txn["amount_log_ratio"])
                    if "amount_log_ratio" in txn
                    else None
                ),
            },
            "customer_history": {
                "prior_transaction_count": int(len(customer_history)),
                "last_prior_transaction": (
                    None
                    if customer_history.empty
                    else str(customer_history.iloc[-1]["transaction_id"])
                ),
            },
            "device_history": {
                "prior_transaction_count": int(len(device_history)),
                "prior_customer_count": int(
                    device_history["customer_id"].nunique()
                ),
            },
            "instrument_history": {
                "prior_transaction_count": int(len(instrument_history)),
                "prior_customer_count": int(
                    instrument_history["customer_id"].nunique()
                ),
                "prior_device_count": int(
                    instrument_history["device_id"].nunique()
                ),
            },
            "graph_evidence": graph,
            "evaluation_only": evaluation,
            "decision": {
                "decision": decision,
                "confidence": confidence,
                "reasons": reasons,
            },
        }

        return evidence, Decision(
            transaction_id=txn_id,
            decision=decision,
            confidence=confidence,
            reasons=reasons,
        )

    @staticmethod
    def to_markdown(evidence: dict) -> str:
        tx = evidence["transaction"]
        ms = evidence["model_signals"]
        ih = evidence["instrument_history"]
        gh = evidence["graph_evidence"]
        dec = evidence["decision"]
        ev = evidence["evaluation_only"]

        lines = [
            f"# Fraud Investigation Case — {tx['transaction_id']}",
            "",
            f"**Decision:** `{dec['decision']}`",
            f"**Confidence:** `{dec['confidence']}`",
            "",
            "## Trigger transaction",
            f"- Customer: `{tx['customer_id']}`",
            f"- Device: `{tx['device_id']}`",
            f"- Instrument: `{tx['instrument_id']}`",
            f"- IP bucket: `{tx['ip_bucket']}`",
            f"- Amount: `{tx['amount']:.0f}` paise",
            f"- ML score: `{tx['ml_score']:.4f}`",
            f"- ML flag: `{tx['ml_flag']}`",
            f"- Graph flag: `{tx['graph_ring_flag']}`",
            f"- Combined flag: `{tx['combined_flag']}`",
            "",
            "## Model evidence",
            f"- velocity_1h: `{ms['velocity_1h']}`",
            f"- velocity_24h: `{ms['velocity_24h']}`",
            f"- time_since_last_txn_sec: `{ms['time_since_last_txn_sec']}`",
            f"- is_new_device: `{ms['is_new_device']}`",
            f"- is_new_instrument: `{ms['is_new_instrument']}`",
            f"- amount_log_ratio: `{ms['amount_log_ratio']}`",
            "",
            "## Historical graph evidence before trigger",
            f"- Prior customers on instrument: `{gh['prior_instrument_customer_count']}`",
            f"- Prior customers: {', '.join(gh['prior_instrument_customers']) or '(none)'}",
            f"- Prior devices on instrument: `{gh['prior_instrument_device_count']}`",
            f"- Prior transactions within 30m: `{gh['prior_transactions_within_30m']}`",
            f"- Prior transaction IDs: {', '.join(gh['prior_transaction_ids_within_30m']) or '(none)'}",
            f"- Prior IP customers: `{gh['prior_ip_customer_count']}`",
            f"- Shared instrument condition: `{gh['shared_instrument_condition']}`",
            f"- Shared IP condition: `{gh['shared_ip_condition']}`",
            f"- Recent burst condition: `{gh['recent_burst_condition']}`",
            f"- Reconstructed graph flag: `{gh['reconstructed_graph_flag']}`",
            "",
            "## Investigator reasoning",
        ]

        if dec["reasons"]:
            lines.extend(f"- {reason}" for reason in dec["reasons"])
        else:
            lines.append("- No strong escalation reason found.")

        lines.extend([
            "",
            "## Evaluation metadata (NOT used for investigation)",
            f"- is_fraud: `{ev['is_fraud_evaluation_only']}`",
            f"- fraud_type: `{ev['fraud_type_evaluation_only']}`",
            "",
            "_This case file is an investigation artifact. It does not alter "
            "the real-time detector decision or introduce future information "
            "into the causal graph evidence._",
        ])

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction", help="Specific transaction ID to investigate.")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Investigate first K combined-flagged transactions.")
    parser.add_argument(
        "--detector",
        default="../features/unified_detector_test.csv",
    )
    parser.add_argument(
        "--history",
        default="../features/features.csv",
    )
    parser.add_argument(
        "--output",
        default="../features/investigation_cases",
    )
    args = parser.parse_args()

    if not args.transaction and args.top_k <= 0:
        parser.error("Provide --transaction or --top-k.")

    investigator = FraudInvestigator(args.detector, args.history)

    if args.transaction:
        txn_ids = [args.transaction]
    else:
        txn_ids = (
            investigator.detector[
                investigator.detector["combined_flag"].eq(1)
            ]
            .sort_values(["ml_score", "timestamp"], ascending=[False, True])
            .head(args.top_k)["transaction_id"]
            .astype(str)
            .tolist()
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for txn_id in txn_ids:
        evidence, decision = investigator.investigate(txn_id)

        md_path = output / f"case_{txn_id}.md"
        json_path = output / f"case_{txn_id}.json"

        md_path.write_text(
            investigator.to_markdown(evidence),
            encoding="utf-8",
        )
        json_path.write_text(
            json.dumps(evidence, indent=2),
            encoding="utf-8",
        )

        print(
            f"{txn_id}: {decision.decision} "
            f"(confidence={decision.confidence})"
        )
        print(f"  Markdown: {md_path}")
        print(f"  JSON:     {json_path}")


if __name__ == "__main__":
    main()