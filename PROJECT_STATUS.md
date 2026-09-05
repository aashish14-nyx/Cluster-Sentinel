Cluster Sentinel — Final Status

Frozen:

Synthetic dataset
Causal feature engineering
Baseline rules
XGBoost model
Time-based validation
Cost-sensitive threshold
Causal graph detector
Graph investigation report
Deterministic investigator
Gemini investigator design
Final headline detector metrics

XGBoost
Precision = 0.669
Recall    = 0.876
F1        = 0.759
PR-AUC    = 0.917
ROC-AUC   = 1.000

Causal graph
Precision = 1.000
Recall    = 0.080
F1        = 0.148

Ring recall = 9/21 = 0.429

LLM evaluation status

Targeted v2 evaluation:

Fraud typology matches = 7/9
Legitimate safely handled = 4/6
Overall targeted checks = 11/15

A later 15-case v3 run was stopped by the Gemini free-tier request quota. No v3 metric is claimed.

Known cleanup

Remove the \i SyntaxWarning from the Windows-path example in agents/llm_investigator.py.

Review the repository for any obsolete retrospective/leaky graph scripts. Do not delete historical experiments blindly; either remove them from the final runtime path or clearly label them as non-causal experiments.

Add requirements.txt.

Keep .env out of version control.