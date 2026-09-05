"""
Cluster Sentinel - Gemini Investigator Evaluation v3

Evaluates the current LLM investigator after adding the explicit
"no_supported_fraud_hypothesis" option.

The true fraud labels are used ONLY after Gemini returns its response.
They are never sent to Gemini.

Run:
    python evaluate_llm_investigator_v3.py
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from fraud_investigator import FraudInvestigator
from llm_investigator import InvestigationResponse, run_gemini


FRAUD_TYPES = [
    "velocity_abuse",
    "device_takeover",
    "shared_instrument_ring",
]

MODEL_DEFAULT = "gemini-3.6-flash"


def choose_cases(detector):
    alerts = detector[detector["combined_flag"].eq(1)].copy()
    selected = []

    for fraud_type in FRAUD_TYPES:
        candidates = alerts[
            alerts["fraud_type"].eq(fraud_type)
        ].sort_values(
            ["ml_score", "timestamp", "transaction_id"],
            ascending=[False, True, True],
        )

        if len(candidates) >= 3:
            positions = [0, len(candidates) // 2, len(candidates) - 1]
            selected.extend(candidates.iloc[positions].to_dict("records"))
        else:
            selected.extend(candidates.to_dict("records"))

    legitimate = alerts[
        alerts["is_fraud"].eq(0)
    ].sort_values(
        ["ml_score", "timestamp", "transaction_id"],
        ascending=[False, True, True],
    )

    if len(legitimate) >= 6:
        positions = [
            0,
            len(legitimate) // 5,
            (2 * len(legitimate)) // 5,
            (3 * len(legitimate)) // 5,
            (4 * len(legitimate)) // 5,
            len(legitimate) - 1,
        ]
        selected.extend(
            legitimate.iloc[positions].to_dict("records")
        )
    else:
        selected.extend(legitimate.to_dict("records"))

    return (
        pd.DataFrame(selected)
        .drop_duplicates(subset=["transaction_id"])
        .sort_values(
            ["is_fraud", "fraud_type", "transaction_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def evaluate_one(investigator, row, model):
    txn_id = str(row["transaction_id"])
    case, _ = investigator.investigate(txn_id)

    result: InvestigationResponse = run_gemini(case, model)

    actual = (
        str(row["fraud_type"])
        if int(row["is_fraud"]) == 1
        else "legitimate"
    )

    if actual == "legitimate":
        match = int(
            result.fraud_hypothesis
            in {"mixed_or_uncertain", "no_supported_fraud_hypothesis"}
        )
    else:
        match = int(result.fraud_hypothesis == actual)

    shape_ok = int(
        result.risk_assessment in {"HIGH", "MEDIUM", "LOW"}
        and 1 <= len(result.strongest_evidence) <= 5
        and 1 <= len(result.recommended_actions) <= 5
        and bool(result.analyst_summary.strip())
    )

    return {
        "transaction_id": txn_id,
        "actual_label_evaluation_only": actual,
        "ml_score": float(row["ml_score"]),
        "graph_flag": int(row["graph_ring_flag"]),
        "combined_flag": int(row["combined_flag"]),
        "gemini_risk": result.risk_assessment,
        "gemini_hypothesis": result.fraud_hypothesis,
        "hypothesis_match_evaluation_only": match,
        "legitimate_safely_not_typed": int(
            actual == "legitimate"
            and result.fraud_hypothesis
            in {"mixed_or_uncertain", "no_supported_fraud_hypothesis"}
        ),
        "response_shape_ok": shape_ok,
        "evidence_count": len(result.strongest_evidence),
        "action_count": len(result.recommended_actions),
        "analyst_summary": result.analyst_summary,
    }


def make_markdown(results, model):
    total = len(results)
    fraud = results[
        results["actual_label_evaluation_only"].isin(FRAUD_TYPES)
    ]
    legit = results[
        results["actual_label_evaluation_only"].eq("legitimate")
    ]

    fraud_matches = int(
        fraud["hypothesis_match_evaluation_only"].sum()
    )
    legit_safe = int(
        legit["legitimate_safely_not_typed"].sum()
    )
    overall = int(
        results["hypothesis_match_evaluation_only"].sum()
    )
    shape = int(
        results["response_shape_ok"].sum()
    )

    lines = [
        "# Cluster Sentinel — Gemini Investigator Evaluation v3",
        "",
        f"Model: `{model}`",
        "",
        "## Results",
        "",
        f"- Total cases: **{total}**",
        f"- Fraud-typology matches: **{fraud_matches}/{len(fraud)}**",
        f"- Legitimate cases safely untyped/uncertain: "
        f"**{legit_safe}/{len(legit)}**",
        f"- Overall hypothesis checks passed: **{overall}/{total}**",
        f"- Structured response checks passed: **{shape}/{total}**",
        "",
        "## Case-level results",
        "",
        "| Transaction | Actual label | Gemini risk | Gemini hypothesis | Match | Safe legitimate handling | Shape OK |",
        "|---|---|---|---|---:|---:|---:|",
    ]

    for row in results.itertuples(index=False):
        lines.append(
            f"| `{row.transaction_id}` | "
            f"`{row.actual_label_evaluation_only}` | "
            f"`{row.gemini_risk}` | "
            f"`{row.gemini_hypothesis}` | "
            f"{row.hypothesis_match_evaluation_only} | "
            f"{row.legitimate_safely_not_typed} | "
            f"{row.response_shape_ok} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "This is a targeted qualitative evaluation of the investigation "
        "assistant. It is not a replacement for the XGBoost or causal graph "
        "detector metrics.",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="../features/unified_detector_test.csv")
    parser.add_argument("--history", default="../features/features.csv")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--csv-out", default="../features/investigation_llm_eval_v3.csv")
    parser.add_argument("--md-out", default="../features/investigation_llm_eval_v3.md")
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set.")

    investigator = FraudInvestigator(args.detector, args.history)
    selected = choose_cases(investigator.detector)

    print(f"Selected {len(selected)} evaluation cases.")

    results = []
    for _, row in selected.iterrows():
        item = evaluate_one(investigator, row, args.model)
        results.append(item)
        print(
            f"{item['transaction_id']}: "
            f"actual={item['actual_label_evaluation_only']} | "
            f"hypothesis={item['gemini_hypothesis']} | "
            f"match={item['hypothesis_match_evaluation_only']}"
        )

    results_df = pd.DataFrame(results)

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)

    md_path = Path(args.md_out)
    md_path.write_text(
        make_markdown(results_df, args.model),
        encoding="utf-8",
    )

    print(f"\nWrote evaluation CSV: {csv_path}")
    print(f"Wrote evaluation report: {md_path}")


if __name__ == "__main__":
    main()
    