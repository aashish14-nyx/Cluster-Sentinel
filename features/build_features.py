import argparse
import os

import numpy as np
import pandas as pd


def load_raw(raw_dir):
    customers = pd.read_csv(os.path.join(raw_dir, "customers.csv"))
    devices = pd.read_csv(os.path.join(raw_dir, "devices.csv"))
    instruments = pd.read_csv(os.path.join(raw_dir, "instruments.csv"))
    transactions = pd.read_csv(os.path.join(raw_dir, "transactions.csv"))
    return customers, devices, instruments, transactions


def add_velocity_features(df):
    """Causal customer velocity/gap features using only prior transactions."""
    df = df.sort_values(["customer_id", "timestamp", "transaction_id"]).reset_index(drop=True)

    velocity_1h = np.zeros(len(df), dtype=np.int32)
    velocity_24h = np.zeros(len(df), dtype=np.int32)
    time_since_last = np.zeros(len(df), dtype=np.float64)
    has_previous_txn = np.zeros(len(df), dtype=np.int8)

    for _, group in df.groupby("customer_id", sort=False):
        idx = group.index.to_numpy()
        ts = group["timestamp"].to_numpy(dtype=np.int64)
        left_1h = left_24h = 0

        for i in range(len(ts)):
            while left_1h < i and ts[i] - ts[left_1h] > 3600:
                left_1h += 1
            while left_24h < i and ts[i] - ts[left_24h] > 86400:
                left_24h += 1

            velocity_1h[idx[i]] = i - left_1h
            velocity_24h[idx[i]] = i - left_24h
            if i > 0:
                has_previous_txn[idx[i]] = 1
                time_since_last[idx[i]] = ts[i] - ts[i - 1]

    df["velocity_1h"] = velocity_1h
    df["velocity_24h"] = velocity_24h
    df["time_since_last_txn_sec"] = time_since_last
    df["has_previous_txn"] = has_previous_txn
    return df


def add_amount_features(df, customers):
    """Customer-relative log amount ratio using a fixed customer baseline."""
    avg_lookup = customers.set_index("customer_id")["historical_avg_amount"]
    df["cust_avg_amount"] = df["customer_id"].map(avg_lookup)
    if df["cust_avg_amount"].isna().any():
        raise ValueError("Some transactions have unknown customer_id values.")

    df["amount_log_ratio"] = (
        np.log(df["amount"].clip(lower=1))
        - np.log(df["cust_avg_amount"].clip(lower=1))
    )
    return df


def add_new_device_instrument_flags(df):
    """Flags first device/instrument use for a customer, based only on prior rows."""
    df = df.sort_values(["customer_id", "timestamp", "transaction_id"]).reset_index(drop=True)
    seen_devices = set()
    seen_instruments = set()
    is_new_device = np.zeros(len(df), dtype=np.int8)
    is_new_instrument = np.zeros(len(df), dtype=np.int8)

    for i, row in enumerate(df.itertuples(index=False)):
        dev_key = (row.customer_id, row.device_id)
        inst_key = (row.customer_id, row.instrument_id)
        if dev_key not in seen_devices:
            is_new_device[i] = 1
            seen_devices.add(dev_key)
        if inst_key not in seen_instruments:
            is_new_instrument[i] = 1
            seen_instruments.add(inst_key)

    df["is_new_device"] = is_new_device
    df["is_new_instrument"] = is_new_instrument
    return df


def add_causal_shared_entity_features(df):
    """Distinct customers seen on device/instrument strictly before current txn."""
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    seen_device_customers = {}
    seen_instrument_customers = {}
    device_count = np.zeros(len(df), dtype=np.int32)
    instrument_count = np.zeros(len(df), dtype=np.int32)

    for i, row in enumerate(df.itertuples(index=False)):
        device_users = seen_device_customers.get(row.device_id)
        instrument_users = seen_instrument_customers.get(row.instrument_id)
        device_count[i] = len(device_users) if device_users else 0
        instrument_count[i] = len(instrument_users) if instrument_users else 0

        if device_users is None:
            device_users = set()
            seen_device_customers[row.device_id] = device_users
        device_users.add(row.customer_id)

        if instrument_users is None:
            instrument_users = set()
            seen_instrument_customers[row.instrument_id] = instrument_users
        instrument_users.add(row.customer_id)

    df["device_customer_count_before"] = device_count
    df["instrument_customer_count_before"] = instrument_count
    return df


def build_feature_table(raw_dir):
    customers, devices, instruments, transactions = load_raw(raw_dir)
    df = transactions.copy()
    df = df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    df = add_velocity_features(df)
    df = add_amount_features(df, customers)
    df = add_new_device_instrument_flags(df)
    df = add_causal_shared_entity_features(df)
    return df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


def validate_features(df):
    required = [
        "transaction_id", "customer_id", "device_id", "instrument_id", "timestamp", "amount",
        "is_fraud", "fraud_type", "velocity_1h", "velocity_24h", "amount_log_ratio",
        "is_new_device", "is_new_instrument", "time_since_last_txn_sec", "has_previous_txn",
        "device_customer_count_before", "instrument_customer_count_before",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if not df["transaction_id"].is_unique:
        raise ValueError("Duplicate transaction_id values found.")
    if df[required].isna().any().any():
        raise ValueError("Unexpected missing values found.")
    if not df["velocity_1h"].ge(0).all() or not df["velocity_24h"].ge(0).all():
        raise ValueError("Velocity features cannot be negative.")
    if not df["time_since_last_txn_sec"].ge(0).all():
        raise ValueError("Time gaps cannot be negative.")
    print("✓ Feature validation passed.")


def main():
    parser = argparse.ArgumentParser(description="Cluster Sentinel causal feature engineering")
    parser.add_argument("--raw-dir", type=str, default="../data/raw")
    parser.add_argument("--out", type=str, default="features.csv")
    args = parser.parse_args()

    print("Building causal fraud features...")
    df = build_feature_table(args.raw_dir)
    validate_features(df)
    df.to_csv(args.out, index=False)

    for col in [
        "velocity_1h", "velocity_24h", "amount_log_ratio", "is_new_device",
        "is_new_instrument", "time_since_last_txn_sec", "has_previous_txn",
        "device_customer_count_before", "instrument_customer_count_before",
    ]:
        print(f"  {col}")
    print(f"Wrote {len(df):,} rows x {len(df.columns)} columns to {args.out}")


if __name__ == "__main__":
    main()