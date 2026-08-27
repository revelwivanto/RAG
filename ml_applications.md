# ML Applications — Four Supervised Models for Procurement

## Purpose

This notebook trains four machine learning models that predict numeric outcomes for laptop procurement requests. While a retrieval-based LLM answers questions about *what* (policies, past purchases, specifications), these models answer *how much* and *how likely* — suitability scores, lifespan estimates, cost predictions, and approval odds.

The models run **offline** as part of a broader procurement assistant pipeline. They do not replace the LLM; they provide the LLM with quantitative outputs to explain to a user.

---

## How It Works

### The Four Use Cases

| UC # | Task | Model | Output | Example |
|---|---|---|---|---|
| **UC1** | Laptop-to-role fit | Random Forest | Suitability probability (0–1) | "32 GB is overkill for Design (88% fit)" |
| **UC2** | Expected lifespan | Ridge regression | Months until failure (20–80) | "This machine will last ~54 months in this role" |
| **UC3** | 3-year operating cost | XGBoost | IDR (maintenance, repair, downtime) | "Total cost of ownership: ~3.5M repair + 30M purchase" |
| **UC4** | Approval odds | XGBoost (imbalanced) | Approval probability (0–1) | "70% likely approved" or "likely rejected" |

### Data Flow

```
User request
    ↓
[Extract fields: department, seniority, specs, price, budget]
    ↓
[Engineer features: ratios, spec adequacy, value density]
    ↓
[UC1] [UC2] [UC3] [UC4]  ← Four models in parallel
    ↓
Four numeric predictions
    ↓
LLM narrates: "For a Design Manager, 32GB is over-provisioned (gap +1.5),
              expected to last 54 months, ~3.5M opex, likely approved."
```

---

## Key Design Decisions

### 1. Feature Engineering (Not Raw Inputs)

**Why:** Trees can only split at fixed thresholds. Ratios are learned more easily.

**Features created:**
- `price_premium_ratio`: How much above the historical average is this quote?
- `spec_gap`: Is the machine over- or under-provisioned for this role? (signed distance)
- `budget_utilisation`: Does this purchase eat the remaining budget?
- `wear_exposure`: How harshly will this role use the machine?

### 2. Labels from Latent Scores, Not If/Else Rules

**Why:** A rule like `if ram >= 32: overspec = True` makes the label deterministic, and trees recover it at ~100% accuracy — a number that means nothing.

**Instead:** Labels are sampled from continuous sigmoid/gamma distributions, creating realistic Bayes error.

### 3. Class Weighting for Imbalance (UC4 Only)

**Problem:** 30% of requests are rejected (minority class).

**Solution:** 
- `scale_pos_weight` in XGBoost reweights the loss to penalize minority errors equally.
- No SMOTE — interpolating between one-hot columns is meaningless.
- Threshold tuned on training F1, applied unchanged to test set.

**Result:** 97% PR-AUC (precision-recall area under curve), which matters more than ROC-AUC under imbalance.

### 4. UC2: Ridge Beats XGBoost (Correctly)

**Why:** The true model is linear. Ridge recognizes this; XGBoost wastes capacity approximating a line.

**Lesson:** Always keep a linear baseline. If it wins, report it — don't force a fancy model to win.

### 5. UC3: Operating Cost, Not Total TCO

**Problem:** Total TCO = purchase_price + operating_cost. Since purchase_price is a feature, predicting total TCO scores R² ≈ 0.99 by echoing the input back.

**Solution:** Predict only operating cost. Total TCO = opex + purchase_price.

---

## Example Walkthrough

**User:** "Beli Macbook 32GB untuk Design Manager, budget 35 juta"

**Extracted fields:**
```
department: "Design"
seniority_level: "Manager"
ram_gb: 32
requested_unit_price: 36,000,000 IDR
dept_budget_remaining: 35,000,000 IDR
```

**Model predictions:**
- **UC1:** suitability = 0.57 (acceptable, but overspec)
- **UC2:** lifespan = 54 months (good durability)
- **UC3:** opex = 3.5M IDR (3-year repairs/downtime)
- **UC4:** approval = 0.45 → "likely rejected"

**LLM output:**
> "Untuk Design Manager, 32GB berlebihan. Meski lifespan lama dan maintenance murah, 
> harga Rp 36M melampaui budget Rp 35M. Sebaiknya cari yang lain dengan 16GB."

---

## How to Run

### Training
```python
from ml_models import train_all
bundle = train_all(n_records=6000)  # → models/procurement_models.joblib
```

### Scoring
```python
from ml_models import score_request

result = score_request(
    department="Design",
    seniority_level="Manager",
    ram_gb=32,
    requested_unit_price=36_000_000,
    dept_budget_remaining=35_000_000,
)
# → {
#     "suitability_proba": 0.568,
#     "expected_lifespan_months": 54.2,
#     "expected_opex_idr": 3_500_000,
#     "approval_proba": 0.45,
#     "approval_decision": "likely rejected",
#     "spec_gap": 1.52,
#     "assumed_fields": ["quantity", "build_quality", ...]  # Defaulted to median
# }
```

---

## Performance

| Use Case | Model | Metric | Score |
|---|---|---|---|
| UC1 | RandomForest | PR-AUC | 0.911 |
| UC2 | Ridge | R² | 0.421 |
| UC3 | XGBoost | R² | 0.866 |
| UC4 | XGBoost | PR-AUC | 0.972 |

---

## Limitations

1. **Synthetic data:** Trained on generated data, not historical BNI records. Demonstrates the pipeline; not evidence of real outcomes.

2. **UC3 scope:** Predicts operating cost (repairs, downtime) only, not total cost of ownership.

3. **Defaults:** Missing fields fall back to training median/mode, which may be misleading for specific requests.
