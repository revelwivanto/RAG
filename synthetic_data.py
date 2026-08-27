import glob
import json
import os
import random

import numpy as np
import pandas as pd

INPUT_DIR = os.path.join("data", "parsed_result")
OUTPUT_PATH = os.path.join("data", "synthetic_data.csv")
NUM_SAMPLES = 1000
YEAR_RANGE = (2021, 2025)

# real_price is intentionally excluded here: it is derived from
# requested_unit_price plus vendor/department factors in generate_record(),
# rather than sampled independently, so the label carries a real relationship
# to the features instead of being pure noise.
NUMERIC_FIELDS = [
    "requested_unit_price",
    "historical_avg_price",
    "quantity",
    "vendor_risk_score",
    "dept_budget_remaining",
]
CATEGORICAL_FIELDS = [
    "department",
    "item_category",
    "is_urgent",
    "search_summary.specs_searched",
    "search_summary.cheapest_vendor_found",
    "search_summary.vendor_channel_type",
]


def flatten(record, prefix=""):
    flat = {}
    for key, value in record.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{full_key}."))
        else:
            flat[full_key] = value
    return flat


def load_real_records():
    records = []
    for path in glob.glob(os.path.join(INPUT_DIR, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            records.append(flatten(json.load(f)))
    if not records:
        raise RuntimeError(f"No parsed JSON records found in {INPUT_DIR}")
    return records


def sample_numeric(values):
    values = np.array([v for v in values if v is not None], dtype=float)
    mean, std = values.mean(), values.std()
    if std == 0:
        std = max(abs(mean) * 0.1, 1.0)
    low, high = values.min(), values.max()
    # Sample around the observed distribution, allowing some spread beyond the
    # narrow real-data range since only a handful of source documents exist.
    value = np.random.normal(mean, std * 1.5)
    return float(np.clip(value, low * 0.5, high * 1.5))


def build_price_factors(pool, low=0.85, high=1.15):
    """Assign each distinct categorical value a fixed price-multiplier, so
    real_price consistently trends higher/lower for that value instead of
    being unrelated to it."""
    return {value: random.uniform(low, high) for value in set(pool)}


def generate_record(real_records, record_id, vendor_factors, dept_factors):
    numeric_pools = {f: [r.get(f) for r in real_records] for f in NUMERIC_FIELDS}
    categorical_pools = {f: [r.get(f) for r in real_records if r.get(f) is not None] for f in CATEGORICAL_FIELDS}

    row = {}
    row["request_id"] = f"PO-SYN-{record_id:06d}"
    row["year"] = random.randint(*YEAR_RANGE)

    for field in CATEGORICAL_FIELDS:
        row[field] = random.choice(categorical_pools[field])

    for field in NUMERIC_FIELDS:
        row[field] = round(sample_numeric(numeric_pools[field]), 2)

    row["quantity"] = max(1, int(round(row["quantity"])))
    row["total_amount"] = round(row["requested_unit_price"] * row["quantity"], 2)
    row["price_variance_ratio"] = round(
        (row["requested_unit_price"] - row["historical_avg_price"]) / row["historical_avg_price"], 4
    ) if row["historical_avg_price"] else 0.0

    # real_price is derived from requested_unit_price plus a vendor/department
    # multiplier and a little noise, giving the label an actual relationship
    # to the features instead of being sampled independently of them.
    vendor_factor = vendor_factors[row["search_summary.vendor_channel_type"]]
    dept_factor = dept_factors[row["department"]]
    noise = np.random.normal(1.0, 0.05)
    row["search_summary.real_price"] = round(
        row["requested_unit_price"] * vendor_factor * dept_factor * noise, 2
    )
    row["search_summary.price_savings_vs_requested"] = round(
        row["requested_unit_price"] - row["search_summary.real_price"], 2
    )

    return row


def main():
    random.seed(42)
    np.random.seed(42)

    real_records = load_real_records()
    vendor_factors = build_price_factors([r.get("search_summary.vendor_channel_type") for r in real_records])
    dept_factors = build_price_factors([r.get("department") for r in real_records])

    synthetic_records = [
        generate_record(real_records, i, vendor_factors, dept_factors) for i in range(1, NUM_SAMPLES + 1)
    ]

    df = pd.DataFrame(synthetic_records)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} synthetic records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
