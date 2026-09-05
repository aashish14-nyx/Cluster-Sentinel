# Cluster Sentinel — Gemini Investigator Evaluation v2

Model: `gemini-3.6-flash`

## Evaluation design

Balanced targeted evaluation with evaluation labels withheld from the Gemini prompt.

- Total cases: **15**
- Fraud-type hypothesis matches: **7/9**
- Legitimate cases correctly treated as uncertain: **4/6**
- Overall hypothesis checks passed: **11/15**
- Response-shape checks passed: **15/15**

## Results

| Transaction | Actual label | Gemini risk | Gemini hypothesis | Match | Shape OK |
|---|---|---|---|---:|---:|
| `pay_0000040008` | `legitimate` | `HIGH` | `mixed_or_uncertain` | 1 | 1 |
| `pay_0000071640` | `legitimate` | `HIGH` | `velocity_abuse` | 0 | 1 |
| `pay_0000088325` | `legitimate` | `HIGH` | `velocity_abuse` | 0 | 1 |
| `pay_0000128262` | `legitimate` | `HIGH` | `mixed_or_uncertain` | 1 | 1 |
| `pay_0000212440` | `legitimate` | `HIGH` | `mixed_or_uncertain` | 1 | 1 |
| `pay_0000293364` | `legitimate` | `HIGH` | `mixed_or_uncertain` | 1 | 1 |
| `pay_0000309165` | `device_takeover` | `HIGH` | `device_takeover` | 1 | 1 |
| `pay_0000309178` | `device_takeover` | `HIGH` | `device_takeover` | 1 | 1 |
| `pay_0000309257` | `device_takeover` | `HIGH` | `device_takeover` | 1 | 1 |
| `pay_0000309421` | `shared_instrument_ring` | `HIGH` | `mixed_or_uncertain` | 0 | 1 |
| `pay_0000309422` | `shared_instrument_ring` | `HIGH` | `shared_instrument_ring` | 1 | 1 |
| `pay_0000309530` | `shared_instrument_ring` | `HIGH` | `shared_instrument_ring` | 1 | 1 |
| `pay_0000308975` | `velocity_abuse` | `HIGH` | `mixed_or_uncertain` | 0 | 1 |
| `pay_0000308980` | `velocity_abuse` | `HIGH` | `velocity_abuse` | 1 | 1 |
| `pay_0000309122` | `velocity_abuse` | `HIGH` | `velocity_abuse` | 1 | 1 |

## Important interpretation

This evaluation is intentionally small and targeted. It is useful for checking whether the investigator distinguishes the synthetic fraud typologies and avoids confidently assigning a fraud type to legitimate alerts. It is not a replacement for statistical model evaluation and should not be reported as detector accuracy.