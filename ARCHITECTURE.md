Cluster Sentinel — Architecture

Runtime path

                  +----------------------+
                  | Synthetic generator |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Causal feature build |
                  +----------+-----------+
                             |
                             v
                    +----------------+
                    |    XGBoost     |
                    +-------+--------+
                            |
                +-----------+-----------+
                |                       |
                v                       v
          ML score/flag          Causal graph
                                      flag
                |                       |
                +-----------+-----------+
                            |
                            v
                  +---------------------+
                  | Deterministic       |
                  | investigator        |
                  +----------+----------+
                             |
                             v
                       Evidence JSON
                             |
                             v
                    +------------------+
                    | Gemini analyst   |
                    +--------+---------+
                             |
                             v
              Summary / hypothesis / actions

Causal graph rule

Before scoring transaction t:

prior instrument customers
prior IP customers
prior transactions in 30-minute window

are inspected.

Only after scoring is the current transaction added to state.

This ordering is what prevents future leakage.

Layer responsibilities

Layer                                                         Responsibility

Feature engineering           ->         Create point-in-time transaction features

XGBoost                       ->         Primary transaction risk score

Causal graph                  ->         Network/temporal corroboration

Deterministic investigator    ->         Retrieve auditable evidence

Gemini                        ->         Explain and prioritize evidence

Evaluation harness            ->         Measure detector and investigator behavior

Important boundary:

The Gemini layer never owns the fraud score.

Detector = objective scoring
Investigator = evidence gathering
LLM = evidence interpretation