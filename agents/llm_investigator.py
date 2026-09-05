"""
Cluster Sentinel - Gemini LLM Fraud Investigator v3

The LLM is an investigation assistant, not the fraud detector.

Key improvement over v2:
- Allows an explicit "no_supported_fraud_hypothesis" outcome.
- This prevents forcing legitimate alerts into one of the three fraud
  typologies.
- The LLM must cite observed evidence and uncertainty.
- Evaluation-only labels are redacted before the model sees the case.

Requirements:
    pip install -U google-genai pydantic python-dotenv

Environment:
    GEMINI_API_KEY=...

Usage:
    python llm_investigator.py --case ..\features\investigation_cases\case_pay_0000309422.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


load_dotenv()


class InvestigationResponse(BaseModel):
    risk_assessment: Literal["HIGH", "MEDIUM", "LOW"]
    fraud_hypothesis: Literal[
        "velocity_abuse",
        "device_takeover",
        "shared_instrument_ring",
        "mixed_or_uncertain",
        "no_supported_fraud_hypothesis",
    ]
    strongest_evidence: list[str] = Field(
        description="Up to five concrete observations from the supplied case."
    )
    uncertainty: list[str] = Field(
        description="Evidence gaps or plausible alternative explanations."
    )
    recommended_actions: list[str] = Field(
        description="Up to five concrete next investigation checks."
    )
    analyst_summary: str = Field(
        description="Concise, evidence-grounded analyst summary."
    )


SYSTEM_INSTRUCTION = """
You are the fraud investigation assistant for Cluster Sentinel.

You receive evidence collected by deterministic fraud-detection code.

Your job is to:
1. Summarize observed evidence.
2. Form the most plausible hypothesis only when evidence supports it.
3. Explicitly admit uncertainty when evidence is insufficient.
4. Recommend practical investigator checks.

Rules:
- Use ONLY the supplied evidence.
- Never invent identifiers, events, relationships, or history.
- Never use evaluation labels; they are redacted.
- A detector alert does NOT prove fraud.
- Do not force an alert into a fraud typology.
- If the evidence does not support a specific abuse pattern, use
  "no_supported_fraud_hypothesis".
- "mixed_or_uncertain" is for cases where multiple abuse hypotheses remain
  plausible.
- Give priority to concrete historical graph evidence and causal model
  features over generic statements.
- Do not alter or recalculate ML scores, graph flags, thresholds, or labels.
- Do not issue a definitive legal/fraud verdict.
"""


def load_case(path: Path) -> dict:
    case = json.loads(path.read_text(encoding="utf-8"))

    required = {
        "transaction",
        "model_signals",
        "customer_history",
        "device_history",
        "instrument_history",
        "graph_evidence",
        "evaluation_only",
        "decision",
    }
    missing = required - set(case)
    if missing:
        raise ValueError(f"Case file missing sections: {sorted(missing)}")

    return case


def sanitize_case(case: dict) -> dict:
    clean = json.loads(json.dumps(case))
    clean["evaluation_only"] = {
        "is_fraud_evaluation_only": "<REDACTED>",
        "fraud_type_evaluation_only": "<REDACTED>",
    }
    return clean


def build_prompt(case: dict) -> str:
    return (
        "Investigate this fraud alert using only the supplied evidence.\n\n"
        "Important:\n"
        "- Evaluation labels are redacted.\n"
        "- Do not assume that a flagged transaction is fraudulent.\n"
        "- A legitimate/benign explanation must remain possible.\n\n"
        "CASE JSON:\n"
        f"{json.dumps(sanitize_case(case), indent=2)}"
    )


def run_gemini(case: dict, model_name: str) -> InvestigationResponse:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. Add it to the project .env file."
        )

    client = genai.Client(api_key=api_key)

    interaction = client.interactions.create(
        model=model_name,
        input=build_prompt(case),
        system_instruction=SYSTEM_INSTRUCTION,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": InvestigationResponse.model_json_schema(),
        },
        generation_config={
            "thinking_level": "low",
        },
    )

    if not interaction.output_text:
        raise RuntimeError("Gemini returned an empty investigation.")

    return InvestigationResponse.model_validate_json(
        interaction.output_text
    )


def to_markdown(case: dict, result: InvestigationResponse) -> str:
    tx = case["transaction"]
    detector = case["decision"]
    graph = case["graph_evidence"]

    lines = [
        f"# Gemini Fraud Investigation — {tx['transaction_id']}",
        "",
        "## Detector context",
        f"- ML score: `{tx['ml_score']:.4f}`",
        f"- ML flag: `{tx['ml_flag']}`",
        f"- Graph flag: `{tx['graph_ring_flag']}`",
        f"- Combined flag: `{tx['combined_flag']}`",
        f"- Deterministic investigator decision: `{detector['decision']}`",
        "",
        "## Gemini assessment",
        f"- Risk: **{result.risk_assessment}**",
        f"- Hypothesis: **{result.fraud_hypothesis}**",
        "",
        "### Strongest evidence",
    ]
    lines.extend(f"- {x}" for x in result.strongest_evidence)

    lines.extend(["", "### Uncertainty / evidence gaps"])
    lines.extend(f"- {x}" for x in result.uncertainty)

    lines.extend(["", "### Recommended investigator actions"])
    lines.extend(f"- {x}" for x in result.recommended_actions)

    lines.extend([
        "",
        "### Analyst summary",
        result.analyst_summary,
        "",
        "## Graph evidence snapshot",
        f"- Prior instrument customers: `{graph['prior_instrument_customer_count']}`",
        f"- Prior instrument transactions within 30m: `{graph['prior_transactions_within_30m']}`",
        f"- Shared instrument: `{graph['shared_instrument_condition']}`",
        f"- Recent burst: `{graph['recent_burst_condition']}`",
        "",
        "---",
        "_Gemini provides investigation assistance only. "
        "Detector metrics and flags remain deterministic and are not modified by the LLM._",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", default="gemini-3.6-flash")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    case_path = Path(args.case)
    case = load_case(case_path)
    result = run_gemini(case, args.model)

    output_path = (
        Path(args.output)
        if args.output
        else case_path.with_name(case_path.stem + "_gemini.md")
    )
    output_path.write_text(
        to_markdown(case, result),
        encoding="utf-8",
    )

    print(f"Transaction: {case['transaction']['transaction_id']}")
    print(f"Gemini risk assessment: {result.risk_assessment}")
    print(f"Fraud hypothesis: {result.fraud_hypothesis}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()