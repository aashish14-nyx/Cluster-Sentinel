Cluster Sentinel

Cluster Sentinel is a production-inspired fraud/abuse detection pipeline for three synthetic abuse patterns:

velocity_abuse

device_takeover

shared_instrument_ring

The project combines causal transaction features, cost-sensitive XGBoost, a causal streaming graph signal, deterministic investigation evidence collection, and a Gemini-powered investigation assistant.

Architecture

                         Synthetic transactions
                                  |
                                  v
                       Causal feature engineering
                                  |
                                  v
                         +----------------+
                         |    XGBoost     |
                         | transaction    |
                         | risk detector  |
                         +----------------+
                                  |
                     +------------+------------+
                     |                         |
                     v                         v
              ML alert/score             Causal graph
                                            signal
                     |                         |
                     +------------+------------+
                                  |
                                  v
                       Deterministic investigator
                                  |
                                  v
                           Evidence JSON
                                  |
                                  v
                       Gemini investigation
                                  |
                                  v
                  Analyst summary / next actions

Key design decisions

Causal features

Features are computed chronologically so each transaction only uses information available before that transaction.

Examples:

velocity_1h

velocity_24h

time_since_last_txn_sec

has_previous_txn

amount_log_ratio

is_new_device

is_new_instrument

device_customer_count_before

instrument_customer_count_before

Time-based evaluation

The data is evaluated chronologically using train/validation/test partitions rather than a random split.

This is important for fraud detection because future behavior must not influence the past.

Cost-sensitive threshold selection

The XGBoost validation threshold is selected using:

false-negative cost = 10

false-positive cost = 1

The selected operating threshold is 0.90.

Model result

On the chronological test set:

Metric

XGBoost

Precision

0.669

Recall

0.876

F1

0.759

PR-AUC

0.917

ROC-AUC

1.000

Confusion matrix:

TP = 99
FP = 49
FN = 14
TN = 46,272

Error analysis

The 14 false negatives were:

7 velocity_abuse

6 shared_instrument_ring

1 device_takeover

The missed velocity cases demonstrate a point-in-time limitation: some planted velocity cases did not have enough observable velocity at the exact transaction timestamp.

The missed ring cases are primarily early ring members. Before enough customers have used an instrument, a causal graph cannot know that the instrument will later become a shared ring without using future information.

This is intentional: the project does not use look-ahead evidence to inflate causal detection metrics.

Causal graph layer

The graph layer processes transactions as a stream.

State is inspected before the current transaction is added.

The graph uses shared instrument evidence and recent temporal activity. IP is used only as temporal corroboration and is not treated as a persistent graph edge.

Causal graph test result:

Precision = 1.000
Recall    = 0.080
F1        = 0.148
Ring recall = 9 / 21 = 0.429

At the selected XGBoost threshold, the 9 causal graph flags were already among the XGBoost flags, so the graph did not increase headline transaction-level recall.

Its value is therefore primarily in network evidence and investigation context rather than pretending to be an incremental detector.

Investigation layer

graph/ring_investigation_report.py reconstructs the state available before each graph-triggered transaction.

For each trigger it can show:

prior customers on the instrument

prior devices

prior transactions in the 30-minute window

prior IP customers

shared-instrument condition

recent-burst condition

reconstructed causal graph flag

This separates:

Real-time detection

from:

Post-detection investigation

LLM investigation layer

agents/fraud_investigator.py collects deterministic evidence and writes a structured JSON case.

agents/llm_investigator.py sends that evidence to Gemini.

The LLM is deliberately not the fraud detector. It:

explains evidence

forms a hypothesis

states uncertainty

recommends investigator actions

Evaluation labels are redacted before the Gemini call.

Targeted LLM evaluation

A targeted 15-case evaluation was used to check the investigation assistant.

Earlier v2 results were:

fraud-typology hypothesis match: 7/9

legitimate alerts safely handled: 4/6

overall targeted checks: 11/15

The main observed weakness was over-classifying some legitimate alerts as velocity_abuse. The investigator was therefore updated to allow:

no_supported_fraud_hypothesis

This prevents the assistant from being forced into one of the three planted fraud categories when evidence is insufficient.

The Gemini API free-tier quota was then reached during the broader v3 run, so no larger v3 result is claimed.

Important evaluation rule

An older retrospective graph implementation used the complete dataset, including future test events. Its results must not be reported as causal production-style metrics.

The causal headline metrics in this README come from the prior-state streaming implementation.

Example case flow

pay_0000309422
    |
    +-- XGBoost: high score
    |
    +-- Causal graph:
    |      2 prior instrument customers
    |      2 prior transactions within 30 minutes
    |      shared instrument condition = true
    |      recent burst condition = true
    |
    +-- Investigator:
    |      ESCALATE_HIGH_RISK
    |
    +-- Gemini:
           HIGH
           shared_instrument_ring

Run order

From the project root:

python data/generate_transactions.py --out-dir data/raw
python features/build_features.py --raw-dir data/raw --out features/features.csv
python models/unified_detector_casual.py --features features/features.csv

The standalone XGBoost script can also be run independently:

python models/train_xgboost.py --features features/features.csv

Run the causal investigation report:

python graph/ring_investigation_report.py --input features/unified_detector_test.csv

Create an evidence case:

python agents/fraud_investigator.py --transaction pay_0000309422

Run the Gemini investigator:

python agents/llm_investigator.py --case features/investigation_cases/case_pay_0000309422.json

Security

Never commit .env.

The Gemini API key belongs in:

GEMINI_API_KEY=...

The repository should keep .env in .gitignore.

Limitations

This is a synthetic fraud/abuse detection project. The dataset is designed for causal experimentation and does not represent real payment behavior.

The graph detector is intentionally conservative. Early ring members may be invisible until the network has accumulated enough evidence.

The LLM layer is an investigation assistant, not an automated adjudication system.

Portfolio takeaway

The strongest engineering story is not "XGBoost got a high score."

It is:

causal feature engineering
        +
time-based validation
        +
cost-sensitive thresholding
        +
causal graph state
        +
evidence reconstruction
        +
LLM-assisted investigation

The project explicitly separates real-time detection from post-detection investigation and avoids look-ahead leakage in its reported causal metrics.


