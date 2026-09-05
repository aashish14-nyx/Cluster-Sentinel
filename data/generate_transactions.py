"""Cluster Sentinel - deterministic synthetic fraud transaction generator."""

import argparse
import os
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "food_delivery", "travel", "utilities",
    "fashion", "entertainment", "pharmacy", "fuel", "subscriptions",
]
CHANNELS = ["card", "upi", "netbanking"]
CHANNEL_WEIGHTS = [0.45, 0.45, 0.10]
HOME_GEOGRAPHIES = [
    "Mumbai", "Delhi", "Bangalore", "Howrah", "Kolkata", "Chennai",
    "Pune", "Hyderabad", "Ahmedabad", "Jaipur",
]
FRAUD_RATES = {
    "velocity_abuse": 0.0012,
    "device_takeover": 0.0008,
    "shared_instrument_ring": 0.0005,
}


class SequentialIDGenerator:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.counter = 0

    def get_one(self) -> str:
        value = f"{self.prefix}_{self.counter:010d}"
        self.counter += 1
        return value

    def get_many(self, count: int):
        if count < 0:
            raise ValueError("count must be non-negative")
        start = self.counter
        self.counter += count
        return [f"{self.prefix}_{i:010d}" for i in range(start, self.counter)]


def rng_choice(rng, options, weights=None):
    return rng.choice(options, p=weights)


def split_event_sizes(total_rows, rng, min_size=4, max_size=8):
    if total_rows == 0:
        return []
    if total_rows < min_size:
        return [total_rows]

    target_events = max(1, int(round(total_rows / ((min_size + max_size) / 2))))
    target_events = min(target_events, total_rows // min_size)
    sizes = [min_size] * target_events
    remaining = total_rows - min_size * target_events

    while remaining:
        candidates = [i for i, s in enumerate(sizes) if s < max_size]
        if not candidates:
            sizes.append(min_size)
            candidates = [len(sizes) - 1]
        idx = int(rng.choice(candidates))
        sizes[idx] += 1
        remaining -= 1

    rng.shuffle(sizes)
    return sizes


def generate_customers(n, rng, start_date, user_gen):
    rows = []
    for _ in range(n):
        signup_offset = int(rng.integers(30, 900))
        signup_date = start_date - timedelta(days=signup_offset)
        historical_avg_amount = int(np.exp(rng.normal(7.0, 0.9)) * 100)
        historical_avg_amount = max(historical_avg_amount, 1000)
        rows.append({
            "customer_id": user_gen.get_one(),
            "signup_date": signup_date.date().isoformat(),
            "home_geography": rng_choice(rng, HOME_GEOGRAPHIES),
            "historical_avg_amount": historical_avg_amount,
            "historical_txn_count": int(rng.integers(5, 500)),
        })
    return pd.DataFrame(rows)


def generate_devices_instruments(customers_df, rng, dev_gen, inst_gen):
    devices, instruments = [], []
    user_devices = {}
    user_instruments = {}

    for cust_id, signup_date in zip(customers_df["customer_id"], customers_df["signup_date"]):
        signup_dt = datetime.fromisoformat(signup_date)

        n_devices = int(rng.choice([1, 2], p=[0.85, 0.15]))
        devs = dev_gen.get_many(n_devices)
        user_devices[cust_id] = devs
        for device_id in devs:
            devices.append({"device_id": device_id, "first_seen": signup_dt.date().isoformat(),
                             "associated_customer_ids": cust_id})

        n_instruments = int(rng.choice([1, 2], p=[0.85, 0.15]))
        insts = inst_gen.get_many(n_instruments)
        user_instruments[cust_id] = insts
        for instrument_id in insts:
            instruments.append({"instrument_id": instrument_id, "first_seen": signup_dt.date().isoformat(),
                                 "associated_customer_ids": cust_id})

    return pd.DataFrame(devices), pd.DataFrame(instruments), user_devices, user_instruments


def generate_normal_transactions(customers_df, rng, user_devices, user_instruments, tx_gen, start_date, days):
    hour_weights = np.array([
        0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0,
        1.5, 2.0, 2.2, 2.3, 2.5, 2.3, 2.0, 2.0,
        2.2, 2.5, 2.8, 2.6, 2.2, 1.8, 1.2, 0.6,
    ])
    hour_weights /= hour_weights.sum()

    rows = []
    for cust in customers_df.itertuples(index=False):
        cust_id = cust.customer_id
        avg_amount = int(cust.historical_avg_amount)
        n_txns = rng.poisson(max(cust.historical_txn_count / 365.0 * days, 0.1))

        devices = user_devices[cust_id]
        instruments = user_instruments[cust_id]

        for _ in range(int(n_txns)):
            day_offset = int(rng.integers(0, days))
            hour = int(rng.choice(24, p=hour_weights))
            minute = int(rng.integers(0, 60))
            ts = start_date + timedelta(days=day_offset, hours=hour, minutes=minute,
                                        seconds=int(rng.integers(0, 60)))

            amount = int(max(np.exp(rng.normal(np.log(max(avg_amount, 1000)), 0.65)), 1000))

            rows.append({
                "transaction_id": tx_gen.get_one(),
                "customer_id": cust_id,
                "device_id": rng_choice(rng, devices),
                "ip_bucket": f"ip_{int(rng.integers(1, 50001)):05d}",
                "instrument_id": rng_choice(rng, instruments),
                "amount": amount,
                "currency": "INR",
                "merchant_category": rng_choice(rng, MERCHANT_CATEGORIES),
                "timestamp": int(ts.timestamp()),
                "channel": rng_choice(rng, CHANNELS, CHANNEL_WEIGHTS),
                "status": "captured",
                "is_fraud": False,
                "fraud_type": "none",
            })

    return pd.DataFrame(rows)


def append_velocity_fraud(rows, customers_df, rng, user_devices, user_instruments, tx_gen,
                          start_date, days, n_rows):
    event_sizes = split_event_sizes(n_rows, rng, min_size=4, max_size=8)
    customer_ids = customers_df["customer_id"].to_numpy()
    selected_users = rng.choice(customer_ids, size=len(event_sizes), replace=False)
    avg_lookup = customers_df.set_index("customer_id")["historical_avg_amount"].to_dict()

    for cust_id, event_size in zip(selected_users, event_sizes):
        device_id = rng.choice(user_devices[cust_id])
        instrument_id = rng.choice(user_instruments[cust_id])
        avg_amount = avg_lookup[cust_id]

        start_offset = int(rng.integers(0, max(1, days * 86400 - 2700)))
        burst_start = start_date + timedelta(seconds=start_offset)
        gaps = rng.exponential(scale=18.0, size=event_size - 1)
        offsets = np.concatenate(([0.0], np.cumsum(gaps)))
        if offsets[-1] > 2400:
            offsets *= 2400.0 / offsets[-1]

        for j, offset in enumerate(offsets):
            small_amount = int(np.clip(
                avg_amount * np.exp(rng.normal(np.log(0.08), 0.65)),
                1000, 30000,
            ))
            ts = burst_start + timedelta(seconds=float(offset))
            rows.append({
                "transaction_id": tx_gen.get_one(),
                "customer_id": cust_id,
                "device_id": device_id,
                "ip_bucket": f"ip_{int(rng.integers(1, 50001)):05d}",
                "instrument_id": instrument_id,
                "amount": small_amount,
                "currency": "INR",
                "merchant_category": rng_choice(rng, MERCHANT_CATEGORIES),
                "timestamp": int(ts.timestamp()),
                "channel": "card",
                "status": "captured",
                "is_fraud": True,
                "fraud_type": "velocity_abuse",
            })


def append_takeover_fraud(rows, new_devices, new_instruments, customers_df, rng, user_devices, user_instruments,
                          tx_gen, dev_gen, inst_gen, start_date, days, n_rows):
    customer_ids = customers_df["customer_id"].to_numpy()
    selected_users = rng.choice(customer_ids, size=min(n_rows, len(customer_ids)), replace=False)
    avg_lookup = customers_df.set_index("customer_id")["historical_avg_amount"].to_dict()

    for cust_id in selected_users:
        ts = start_date + timedelta(
            days=int(rng.integers(0, days)), hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)),
        )
        new_device = dev_gen.get_one()
        new_devices.append({"device_id": new_device, "first_seen": ts.date().isoformat(),
                             "associated_customer_ids": cust_id})

        # Guide's typology is "never-seen device AND instrument together" - mint both.
        new_instrument = inst_gen.get_one()
        new_instruments.append({"instrument_id": new_instrument, "first_seen": ts.date().isoformat(),
                                 "associated_customer_ids": cust_id})

        amount = int(np.clip(avg_lookup[cust_id] * rng.uniform(2.0, 6.0), 2000, 1_500_000))
        rows.append({
            "transaction_id": tx_gen.get_one(),
            "customer_id": cust_id,
            "device_id": new_device,
            "ip_bucket": f"ip_{int(rng.integers(1, 50001)):05d}",
            "instrument_id": new_instrument,
            "amount": amount,
            "currency": "INR",
            "merchant_category": rng_choice(rng, MERCHANT_CATEGORIES),
            "timestamp": int(ts.timestamp()),
            "channel": rng.choice(["mobile_app", "web_browser"], p=[0.35, 0.65]),
            "status": "captured",
            "is_fraud": True,
            "fraud_type": "device_takeover",
        })


def append_ring_fraud(rows, new_instruments, customers_df, rng, user_devices,
                      tx_gen, inst_gen, start_date, days, n_rows):
    if n_rows == 0:
        return

    customer_ids = customers_df["customer_id"].to_numpy()
    avg_lookup = customers_df.set_index("customer_id")["historical_avg_amount"].to_dict()
    ring_sizes = split_event_sizes(n_rows, rng, min_size=3, max_size=5)

    for ring_size in ring_sizes:
        members = rng.choice(customer_ids, size=ring_size, replace=False).tolist()
        shared_instrument = inst_gen.get_one()
        shared_ip = f"ip_{int(rng.integers(1, 50001)):05d}"
        burst_start = start_date + timedelta(days=int(rng.integers(0, days)), hours=int(rng.integers(0, 24)))

        new_instruments.append({"instrument_id": shared_instrument, "first_seen": burst_start.date().isoformat(),
                                 "associated_customer_ids": "|".join(members)})

        for j, cust_id in enumerate(members):
            ts = burst_start + timedelta(minutes=int(rng.integers(2, 12)) * (j + 1))
            amount = int(np.clip(avg_lookup[cust_id] * rng.uniform(0.8, 3.5), 2000, 750_000))
            rows.append({
                "transaction_id": tx_gen.get_one(),
                "customer_id": cust_id,
                "device_id": rng.choice(user_devices[cust_id]),
                "ip_bucket": shared_ip,
                "instrument_id": shared_instrument,
                "amount": amount,
                "currency": "INR",
                "merchant_category": rng_choice(rng, MERCHANT_CATEGORIES),
                "timestamp": int(ts.timestamp()),
                "channel": rng_choice(rng, CHANNELS, CHANNEL_WEIGHTS),
                "status": "captured",
                "is_fraud": True,
                "fraud_type": "shared_instrument_ring",
            })


def validate_dataset(transactions, customers, devices, instruments,
                     expected_fraud_ratio, expected_counts, start, end):
    assert transactions["transaction_id"].is_unique
    assert customers["customer_id"].is_unique
    assert devices["device_id"].is_unique
    assert instruments["instrument_id"].is_unique
    assert transactions["customer_id"].isin(customers["customer_id"]).all()
    assert transactions["device_id"].isin(devices["device_id"]).all()
    assert transactions["instrument_id"].isin(instruments["instrument_id"]).all()
    assert (transactions["amount"] > 0).all()
    assert transactions.notna().all().all(), "Unexpected missing values in generated dataset"

    ts = pd.to_datetime(transactions["timestamp"], unit="s", utc=True)
    assert ts.ge(start).all()
    assert ts.lt(end).all()

    actual_ratio = transactions["is_fraud"].mean()
    tolerance = max(1e-5, 3.0 / len(transactions))
    assert abs(actual_ratio - expected_fraud_ratio) <= tolerance, (
        f"Fraud ratio mismatch: {actual_ratio:.6%} vs {expected_fraud_ratio:.6%} "
        f"(tolerance {tolerance:.6%})"
    )

    counts = transactions.loc[transactions["is_fraud"], "fraud_type"].value_counts().to_dict()
    for fraud_type, expected in expected_counts.items():
        assert counts.get(fraud_type, 0) == expected, (fraud_type, counts, expected_counts)

    velocity = transactions[transactions["fraud_type"] == "velocity_abuse"]
    if not velocity.empty:
        min_expected_users = max(1, len(velocity) // 8)
        assert velocity["customer_id"].nunique() >= min_expected_users, "Velocity fraud concentrated in too few users"

    ring = transactions[transactions["fraud_type"] == "shared_instrument_ring"]
    if not ring.empty:
        ring_users = ring.groupby("instrument_id")["customer_id"].nunique()
        assert (ring_users >= 3).all(), "Every ring instrument must connect >=3 customers"
        assert ring["instrument_id"].nunique() >= 2 or len(ring) < 6, "Ring fraud collapsed into one ring"


def build_dataset(seed=42, n_customers=5000, days=90, target_fraud_ratio=0.0025):
    if n_customers < 10:
        raise ValueError("n_customers must be >= 10")
    if days <= 0:
        raise ValueError("days must be positive")
    if not 0 < target_fraud_ratio < 1:
        raise ValueError("target_fraud_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    # Explicit UTC avoids a cross-platform bug: datetime.timestamp() on a
    # naive datetime silently uses the local system timezone, which makes
    # generated data non-reproducible (and can fail validation) on any
    # machine that isn't set to UTC.
    start_dt = datetime(2026, 6, 1, tzinfo=timezone.utc)
    start = pd.Timestamp(start_dt)
    end = start + pd.Timedelta(days=days)

    user_gen = SequentialIDGenerator("cust")
    dev_gen = SequentialIDGenerator("dev")
    inst_gen = SequentialIDGenerator("inst")
    tx_gen = SequentialIDGenerator("pay")

    customers = generate_customers(n_customers, rng, start_dt, user_gen)
    devices, instruments, user_devices, user_instruments = generate_devices_instruments(
        customers, rng, dev_gen, inst_gen
    )
    normal = generate_normal_transactions(customers, rng, user_devices, user_instruments, tx_gen, start_dt, days)

    fraud_total = int(round(len(normal) * target_fraud_ratio / (1 - target_fraud_ratio)))
    n_velocity = int(round(fraud_total * 0.48))
    n_takeover = int(round(fraud_total * 0.32))
    n_ring = fraud_total - n_velocity - n_takeover

    fraud_rows = []
    takeover_devices = []
    takeover_instruments = []
    ring_instruments = []

    append_velocity_fraud(fraud_rows, customers, rng, user_devices, user_instruments, tx_gen, start_dt, days, n_velocity)
    append_takeover_fraud(fraud_rows, takeover_devices, takeover_instruments, customers, rng, user_devices, user_instruments,
                          tx_gen, dev_gen, inst_gen, start_dt, days, n_takeover)
    append_ring_fraud(fraud_rows, ring_instruments, customers, rng, user_devices, tx_gen, inst_gen, start_dt, days, n_ring)

    fraud_df = pd.DataFrame(fraud_rows)
    transactions = pd.concat([normal, fraud_df], ignore_index=True)
    transactions = transactions.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    if takeover_devices:
        devices = pd.concat([devices, pd.DataFrame(takeover_devices)], ignore_index=True)
    if takeover_instruments:
        instruments = pd.concat([instruments, pd.DataFrame(takeover_instruments)], ignore_index=True)
    if ring_instruments:
        instruments = pd.concat([instruments, pd.DataFrame(ring_instruments)], ignore_index=True)

    expected_counts = {"velocity_abuse": n_velocity, "device_takeover": n_takeover, "shared_instrument_ring": n_ring}
    validate_dataset(transactions, customers, devices, instruments, target_fraud_ratio, expected_counts, start, end)

    return transactions, customers, devices, instruments, expected_counts


def main():
    parser = argparse.ArgumentParser(description="Cluster Sentinel synthetic fraud generator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-customers", type=int, default=5000)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--fraud-ratio", type=float, default=0.0025)
    parser.add_argument("--out-dir", type=str, default="raw")
    args = parser.parse_args()

    transactions, customers, devices, instruments, counts = build_dataset(
        seed=args.seed, n_customers=args.n_customers, days=args.days, target_fraud_ratio=args.fraud_ratio,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    customers.to_csv(os.path.join(args.out_dir, "customers.csv"), index=False)
    devices.to_csv(os.path.join(args.out_dir, "devices.csv"), index=False)
    instruments.to_csv(os.path.join(args.out_dir, "instruments.csv"), index=False)
    transactions.to_csv(os.path.join(args.out_dir, "transactions.csv"), index=False)

    print(f"Rows: {len(transactions):,}")
    print(f"Fraud: {int(transactions['is_fraud'].sum()):,} ({transactions['is_fraud'].mean():.3%})")
    print(f"Fraud counts: {counts}")
    print(f"Velocity users: {transactions.loc[transactions.fraud_type == 'velocity_abuse', 'customer_id'].nunique()}")
    print(f"Ring instruments: {transactions.loc[transactions.fraud_type == 'shared_instrument_ring', 'instrument_id'].nunique()}")
    print(f"Output: {args.out_dir}")


if __name__ == "__main__":
    main()