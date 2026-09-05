# Cluster Sentinel — Spec

## Transaction schema
transaction_id      -- e.g. pay_DESp9bgForNoUd (Razorpay's real id format)
customer_id          -- derived from email/contact
device_id             -- synthetic, not provided by Razorpay
ip_bucket             -- synthetic, not provided by Razorpay
instrument_id        -- maps to Razorpay's card_id / vpa
amount                -- in paise (int), matches Razorpay convention
currency              -- INR
merchant_category  -- synthetic (Razorpay doesn't expose this)
timestamp            -- unix time, matches created_at
channel               -- card / upi / netbanking (from method field)
status                 -- authorized / captured / failed
is_fraud               -- ground-truth label, held out at inference

## Customer schema
customer_id, signup_date, home_geography, historical_avg_amount, historical_txn_count

## Device/Instrument schema
device_id / instrument_id, first_seen, associated_customer_ids

## Fraud typologies injected (pick 2-3 from guide §5)
1. Velocity abuse — target rate: 0.12%
2. New-device/instrument takeover — target rate: 0.08%
3. Shared-instrument ring -- target rate: 0.05%

## Overall fraud prevalence target
0.1–0.3% (matches ULB/PaySim real-world rates)