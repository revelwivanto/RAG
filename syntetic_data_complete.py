"""Synthetic procurement dataset for the four ML use cases in ml_applications.md.

Design notes
------------
The labels are *not* hard if/else rules. Each one is built as a latent score
from the features, pushed through a sigmoid (classification) or used as a
distribution mean (regression), and then sampled. That matters: a rule like
`if ram < 16: match = 0` makes the label a lookup table, so a tree model just
rediscovers the threshold and reports near-perfect accuracy that means nothing.
Sampling from a latent score leaves genuine Bayes error, so the metrics in
ml_applications.ipynb reflect a real decision surface.

The feature set also covers what ml_applications.md actually asks for --
travel frequency (UC2), per-model IT ticket history (UC3), and time since last
upgrade (UC4) -- which the first version of this script promised but never
generated.

FEATURE_COLUMNS and TARGET_COLUMNS are exported so the training notebook can
build X without ever touching another use case's label. All four targets live
in one table, so cross-target leakage is the easiest mistake to make here.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------

# compute_tier   1 = office/browser, 2 = general, 3 = heavy, 4 = workstation
# build_quality  0..1, drives how well the chassis survives being carried around
# portability    0..1, higher = lighter/more travel-friendly
LAPTOPS = [
    # brand,   model,            cpu,        gpu,          base_ram, base_ssd, base_price, tier, build, port
    ("Asus",   "ROG Zephyrus",   "Core i9",  "RTX 4070",   32,  1024, 35_000_000, 4, 0.55, 0.30),
    ("Asus",   "ExpertBook B9",  "Core i7",  "Iris Xe",    16,  1024, 25_000_000, 2, 0.80, 0.95),
    ("Asus",   "Vivobook 14",    "Core i5",  "Integrated",  8,   512, 10_500_000, 1, 0.50, 0.70),
    ("Lenovo", "ThinkPad P16",   "Core i9",  "RTX A2000",  32,  1024, 38_000_000, 4, 0.90, 0.25),
    ("Lenovo", "ThinkPad T14",   "Core i5",  "Integrated", 16,   512, 18_000_000, 2, 0.92, 0.75),
    ("Lenovo", "ThinkPad X1",    "Core i7",  "Iris Xe",    16,  1024, 28_000_000, 3, 0.90, 0.95),
    ("Lenovo", "IdeaPad Slim 3", "Core i3",  "Integrated",  8,   256,  8_000_000, 1, 0.45, 0.65),
    ("HP",     "ZBook Firefly",  "Core i7",  "RTX A500",   32,  1024, 30_000_000, 3, 0.80, 0.60),
    ("HP",     "OmniBook 5",     "Ryzen 7",  "Radeon",     16,   512, 17_500_000, 2, 0.65, 0.80),
    ("HP",     "ProBook 445",    "Ryzen 5",  "Radeon",      8,   512, 12_000_000, 1, 0.60, 0.70),
    ("Dell",   "Precision 5690", "Core i9",  "RTX 3500",   32,  2048, 42_000_000, 4, 0.85, 0.35),
    ("Dell",   "Latitude 7450",  "Core i7",  "Integrated", 16,   512, 22_000_000, 2, 0.85, 0.85),
    ("Dell",   "Inspiron 15",    "Core i5",  "Integrated",  8,   512,  9_500_000, 1, 0.50, 0.60),
    ("Apple",  "MacBook Pro 14", "M3 Pro",   "Apple GPU",  18,   512, 32_000_000, 4, 0.95, 0.85),
    ("Apple",  "MacBook Air 13", "M3",       "Apple GPU",   8,   256, 18_500_000, 2, 0.95, 1.00),
]

LAPTOP_COLUMNS = [
    "laptop_brand", "laptop_model", "cpu", "gpu",
    "base_ram_gb", "base_storage_gb", "base_price", "compute_tier",
    "build_quality", "portability",
]

# compute_need   what the role actually needs, on the same 1..4 scale as compute_tier
# travel         baseline trips per month, drives wear in UC2
# windows_locked share of the department that cannot use macOS (Excel macros, core banking apps)
DEPARTMENTS = {
    "IT":            {"compute_need": 3.4, "travel": 0.8, "windows_locked": 0.35, "budget_cap": 32_000_000},
    "Design":        {"compute_need": 3.6, "travel": 0.6, "windows_locked": 0.10, "budget_cap": 30_000_000},
    "Finance":       {"compute_need": 2.0, "travel": 0.7, "windows_locked": 0.90, "budget_cap": 20_000_000},
    "HR":            {"compute_need": 1.5, "travel": 0.5, "windows_locked": 0.60, "budget_cap": 15_000_000},
    "Sales":         {"compute_need": 1.8, "travel": 4.5, "windows_locked": 0.55, "budget_cap": 18_000_000},
    "Medical Staff": {"compute_need": 2.2, "travel": 1.2, "windows_locked": 0.70, "budget_cap": 22_000_000},
}

# prestige lifts what a role is expected to receive; multiplier scales the policy cap
SENIORITY = {
    "Staff":    {"prestige": 0.0, "cap_multiplier": 0.75},
    "Senior":   {"prestige": 0.6, "cap_multiplier": 1.00},
    "Manager":  {"prestige": 1.3, "cap_multiplier": 1.35},
    "Director": {"prestige": 2.0, "cap_multiplier": 1.90},
}

RAM_UPGRADES = [0, 8, 16]        # GB added on top of the base configuration
STORAGE_UPGRADES = [0, 512, 1024]
WARRANTY_YEARS = [1, 2, 3]

# Rough street cost of an upgrade, per unit, in IDR.
RAM_COST_PER_GB = 250_000
STORAGE_COST_PER_GB = 4_000


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_procurement_ml_data(num_records=4000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    catalogue = pd.DataFrame(LAPTOPS, columns=LAPTOP_COLUMNS)

    # Per-model reliability that the buyer cannot see directly. UC3's ticket
    # history is a noisy observation of it, which is what makes UC3 learnable
    # without simply handing over the answer.
    model_defect_rate = {
        row.laptop_model: float(np.clip(rng.normal(0.30 - 0.22 * row.build_quality, 0.05), 0.03, 0.60))
        for row in catalogue.itertuples()
    }

    picks = rng.integers(0, len(catalogue), size=num_records)
    rows = []

    for i in range(num_records):
        spec = catalogue.iloc[picks[i]]

        dept = rng.choice(list(DEPARTMENTS))
        dept_cfg = DEPARTMENTS[dept]
        role = rng.choice(list(SENIORITY), p=[0.45, 0.30, 0.18, 0.07])
        role_cfg = SENIORITY[role]

        # ---------------- configuration actually ordered ----------------
        ram = int(spec.base_ram_gb + rng.choice(RAM_UPGRADES, p=[0.60, 0.28, 0.12]))
        storage = int(spec.base_storage_gb + rng.choice(STORAGE_UPGRADES, p=[0.65, 0.25, 0.10]))
        warranty_years = int(rng.choice(WARRANTY_YEARS, p=[0.45, 0.35, 0.20]))

        config_price = (
            spec.base_price
            + (ram - spec.base_ram_gb) * RAM_COST_PER_GB
            + (storage - spec.base_storage_gb) * STORAGE_COST_PER_GB
            + (warranty_years - 1) * 1_200_000
        )

        # Vendors quote around the configured price; the spread is the thing
        # UC4 has to react to, so it is wider than a token +/-10%.
        historical_avg_price = int(config_price * rng.normal(0.97, 0.03))
        requested_unit_price = int(config_price * rng.normal(1.02, 0.09))
        quantity = int(rng.integers(1, 26))

        # ---------------- employee / request context ----------------
        tenure_months = int(np.clip(rng.gamma(2.2, 22), 1, 240))
        months_since_last_upgrade = int(np.clip(rng.normal(34, 14), 1, 96))
        travel_days_per_month = float(np.clip(rng.gamma(2.0, dept_cfg["travel"] / 2.0), 0, 22))
        prior_replacements = int(rng.poisson(0.45))

        vendor_risk_score = float(np.clip(rng.beta(2.2, 4.5), 0.01, 0.99))
        vendor_is_official = int(rng.random() < (0.75 - 0.5 * vendor_risk_score))

        policy_cap = dept_cfg["budget_cap"] * role_cfg["cap_multiplier"]
        # Centred well above the request so most departments can actually afford
        # what they ask for; a tighter sigma keeps wildly-over-budget requests
        # rare rather than routine.
        # The ceiling is high enough not to truncate the lognormal: at a 900M cap
        # roughly a fifth of rows piled up on the boundary as an identical value,
        # which is an artefact rather than a budget.
        dept_budget_remaining = int(np.clip(
            rng.lognormal(np.log(policy_cap * quantity * 1.8), 0.60),
            5_000_000, 6_000_000_000,
        ))
        is_urgent = int(rng.random() < 0.18)
        requires_windows = int(rng.random() < dept_cfg["windows_locked"])

        total_amount = requested_unit_price * quantity

        # ==============================================================
        # UC1 -- suitability. Under-spec hurts a lot, over-spec hurts a
        # little (wasted budget), which is the signal the assistant needs
        # to say "32 GB is overkill for this role".
        # ==============================================================
        capability = spec.compute_tier + 0.55 * np.log2(max(ram, 1) / 8.0)
        gap = capability - dept_cfg["compute_need"]
        fit = -1.55 * max(0.0, -gap) ** 1.5 - 0.42 * max(0.0, gap) ** 1.3

        prestige_gap = role_cfg["prestige"] - (requested_unit_price / 12_000_000.0)

        # Intercept calibrated so roughly 70% of past deployments succeeded --
        # a 30% success rate would not describe any real procurement function.
        uc1_score = (
            3.75
            + 1.5 * fit
            - 0.55 * abs(prestige_gap)
            - 2.1 * (requires_windows and spec.laptop_brand == "Apple")
            + 1.25 * spec.portability * (travel_days_per_month / 10.0)
            - 0.9 * (1.0 - spec.build_quality) * (travel_days_per_month / 10.0)
            + 0.30 * (storage >= 512)
        )
        is_successful_match = int(rng.random() < _sigmoid(uc1_score))

        # ==============================================================
        # UC2 -- months until first major failure. Gamma keeps it positive
        # and right-skewed, which is how hardware survival actually looks.
        # ==============================================================
        # Centred so the fleet mean lands near a ~44-month refresh cycle.
        expected_life = (
            18.0
            + 32.0 * spec.build_quality
            - 1.35 * travel_days_per_month
            - 3.2 * prior_replacements
            + 5.0 * warranty_years
            - 14.0 * model_defect_rate[spec.laptop_model]
        )
        expected_life = float(np.clip(expected_life, 8.0, 84.0))
        # Gamma sd is mean/sqrt(shape). At shape=9 that is ~15 months against a
        # systematic spread of about the same size, leaving R^2 near 0.2 -- the
        # noise drowns the signal. shape=25 puts sd near 9 months, so a model
        # can recover the durability signal while real error remains.
        shape = 25.0
        months_until_failure = float(np.clip(
            rng.gamma(shape, expected_life / shape), 3.0, 120.0,
        ))

        # Observable proxy for reliability: tickets logged against this model
        # across the fleet. Correlated with the hidden defect rate, not equal.
        it_tickets_last_year = int(rng.poisson(
            np.clip(6.0 * model_defect_rate[spec.laptop_model] + 0.12 * travel_days_per_month, 0.1, 12.0)
        ))

        # ==============================================================
        # UC3 -- 3-year total cost of ownership.
        # ==============================================================
        setup_cost = rng.normal(1_500_000, 250_000)
        ecosystem_tax = 3_000_000 if spec.laptop_brand == "Apple" else 0
        # Repairs inside the 36-month window, minus whatever warranty absorbs.
        expected_repairs = max(0.0, (36.0 - months_until_failure) / 36.0) * 2.0
        covered = min(warranty_years, 3) / 3.0
        repair_cost = expected_repairs * 4_200_000 * (1.0 - 0.65 * covered)
        # Lost productivity while a machine is in for service. Scales with the
        # seniority of whoever is idle, which is why it is not a flat rate.
        downtime_rate = rng.normal(420_000, 90_000) * (1.0 + 0.45 * role_cfg["prestige"])
        downtime_cost = it_tickets_last_year * 3 * downtime_rate

        # Everything the purchase price does NOT already tell you.
        opex_3_years_idr = float(setup_cost + ecosystem_tax + repair_cost + downtime_cost)
        tco_3_years_idr = float(requested_unit_price + opex_3_years_idr)

        # ==============================================================
        # UC4 -- approval. Deliberately imbalanced (~20-25% rejected) so the
        # notebook's imbalance handling is doing real work.
        # ==============================================================
        budget_ratio = total_amount / max(dept_budget_remaining, 1)
        price_premium = (requested_unit_price - historical_avg_price) / max(historical_avg_price, 1)
        cap_overrun = (requested_unit_price - policy_cap) / policy_cap

        # Calibrated to ~78% approved. The imbalance is the point: a balanced
        # approval label would make the notebook's class-weighting a no-op.
        uc4_score = (
            4.05
            # Clipped: budget_ratio has a long right tail, and an unclipped
            # hinge sent those rows to a logit near -90, making them
            # deterministic rejections with no Bayes error left to learn.
            - 2.8 * min(max(0.0, budget_ratio - 0.55), 2.5)
            - 3.4 * min(max(0.0, cap_overrun), 1.5)
            - 4.5 * max(0.0, price_premium - 0.05)
            # Linear, not hinged: vendor_risk_score is Beta(2.2, 4.5), so a
            # hinge at 0.55 sat above the 75th percentile and the term was
            # almost always zero -- approval ended up barely reacting to risk.
            - 3.2 * vendor_risk_score
            + 0.55 * vendor_is_official
            + 0.75 * is_urgent
            + 0.014 * months_since_last_upgrade
            + 0.30 * role_cfg["prestige"]
        )
        is_approved = int(rng.random() < _sigmoid(uc4_score))

        rows.append({
            "request_id": f"REQ-2026-{i:05d}",
            # --- request context
            "department": dept,
            "seniority_level": role,
            "employee_tenure_months": tenure_months,
            "months_since_last_upgrade": months_since_last_upgrade,
            "travel_days_per_month": round(travel_days_per_month, 2),
            "prior_replacements": prior_replacements,
            "requires_windows": requires_windows,
            "is_urgent": is_urgent,
            # --- product
            "laptop_brand": spec.laptop_brand,
            "laptop_model": spec.laptop_model,
            "cpu": spec.cpu,
            "gpu": spec.gpu,
            "ram_gb": ram,
            "storage_gb": storage,
            "compute_tier": int(spec.compute_tier),
            "build_quality": round(float(spec.build_quality), 2),
            "portability": round(float(spec.portability), 2),
            "warranty_years": warranty_years,
            "it_tickets_last_year": it_tickets_last_year,
            # --- commercial
            "quantity": quantity,
            "requested_unit_price": requested_unit_price,
            "historical_avg_price": historical_avg_price,
            "total_amount": total_amount,
            "vendor_risk_score": round(vendor_risk_score, 3),
            "vendor_is_official": vendor_is_official,
            "dept_budget_remaining": dept_budget_remaining,
            "dept_policy_cap": int(policy_cap),
            # --- targets
            "target_uc1_is_match": is_successful_match,
            "target_uc2_months_to_failure": round(months_until_failure, 1),
            "target_uc3_tco_idr": round(tco_3_years_idr, 0),
            # The modelling target for UC3. Total TCO is dominated by
            # requested_unit_price, which is a *feature* -- predicting it scores
            # R^2 ~ 0.997 by echoing an input back and tells finance nothing they
            # did not already know. The operating cost is the unknown half.
            "target_uc3_opex_idr": round(opex_3_years_idr, 0),
            "target_uc4_is_approved": is_approved,
        })

    return pd.DataFrame(rows)


# Exported so the notebook can assemble X without ever reaching into another
# use case's label. All four targets share one table, which makes cross-target
# leakage the easiest mistake available here.
TARGET_COLUMNS = [
    "target_uc1_is_match",
    "target_uc2_months_to_failure",
    "target_uc3_tco_idr",
    "target_uc3_opex_idr",
    "target_uc4_is_approved",
]

ID_COLUMNS = ["request_id"]

CATEGORICAL_FEATURES = [
    "department", "seniority_level", "laptop_brand", "laptop_model", "cpu", "gpu",
]

NUMERIC_FEATURES = [
    "employee_tenure_months", "months_since_last_upgrade", "travel_days_per_month",
    "prior_replacements", "requires_windows", "is_urgent",
    "ram_gb", "storage_gb", "compute_tier", "build_quality", "portability",
    "warranty_years", "it_tickets_last_year",
    "quantity", "requested_unit_price", "historical_avg_price", "total_amount",
    "vendor_risk_score", "vendor_is_official", "dept_budget_remaining", "dept_policy_cap",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


if __name__ == "__main__":
    df = generate_procurement_ml_data(4000)
    df.to_csv("laptop_procurement_ml_training_data.csv", index=False)

    print("Dataset shape:", df.shape)
    print("\nLabel balance / distribution:")
    print(f"  UC1 match rate      : {df.target_uc1_is_match.mean():.1%}")
    print(f"  UC2 months to fail  : mean {df.target_uc2_months_to_failure.mean():.1f}, "
          f"sd {df.target_uc2_months_to_failure.std():.1f}")
    print(f"  UC3 TCO (IDR)       : mean {df.target_uc3_tco_idr.mean():,.0f}")
    print(f"  UC3 opex (IDR)      : mean {df.target_uc3_opex_idr.mean():,.0f}, "
          f"sd {df.target_uc3_opex_idr.std():,.0f}")
    print(f"  UC4 approval rate   : {df.target_uc4_is_approved.mean():.1%}")
    print(f"\n{len(FEATURE_COLUMNS)} features, {len(TARGET_COLUMNS)} targets")
