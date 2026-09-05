# Gemini Fraud Investigation — pay_0000309422

## Detector context
- ML score: `1.0000`
- ML flag: `1`
- Graph flag: `1`
- Combined flag: `1`
- Deterministic investigator decision: `ESCALATE_HIGH_RISK`

## Gemini assessment
- Risk assessment: **HIGH**
- Fraud hypothesis: **shared_instrument_ring**

### Strongest evidence
- Instrument inst_0000006008 is brand new to customer cust_0000004431 but has been previously used by 2 other customers (cust_0000001932, cust_0000004273) across 2 distinct devices.
- Two prior transactions (pay_0000309420, pay_0000309421) occurred on the same instrument within 30 minutes of this event.
- IP bucket ip_24251 was previously associated with 4 distinct customers.
- Detector outputs generated high-risk flags (ml_score: 0.9999535, graph_ring_flag: 1, combined_flag: 1).
- High transaction amount of 567359.0 using a newly attached shared payment instrument.

### Uncertainty / evidence gaps
- Whether the payment instrument is a legitimate shared/corporate funding source or authorized secondary card.
- Lack of identity verification data or KYC details confirming relationships between the linked customer accounts.

### Recommended investigator actions
- Verify ownership and authorization for payment instrument inst_0000006008 across linked customers cust_0000001932 and cust_0000004273.
- Inspect the timing and activity of recent burst transactions pay_0000309420 and pay_0000309421.
- Review IP bucket ip_24251 for potential proxy usage or shared network infrastructure.
- Contact customer cust_0000004431 to confirm the high-value 567359.0 transaction.

### Analyst summary
Transaction pay_0000309422 represents a high-risk event flagged by both machine learning (score 0.9999535) and graph ring detectors. Customer cust_0000004431 attempted a high-value transaction of 567359.0 using instrument inst_0000006008, which is new to this customer but linked to two other distinct customers and devices. Furthermore, two burst transactions were recorded on the instrument within 30 minutes, and the IP bucket is shared among four customers. This cluster pattern strongly indicates a shared instrument fraud ring. Immediate review of instrument authorization and account linkings is recommended.

## Key structured evidence
- velocity_1h: `0`
- velocity_24h: `0`
- is_new_device: `0`
- is_new_instrument: `1`
- Prior instrument customers: `2`
- Prior instrument transactions in 30m: `2`

---
_Gemini is an investigation assistant only. The deterministic detector remains the source of risk scoring and graph signaling. Evaluation labels are excluded from the LLM prompt._