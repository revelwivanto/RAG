"""Train, persist and serve the four procurement models.

ml_applications.ipynb explores these -- feature engineering, the leakage guard,
hyperparameter search, and the metric comparison that picked each winner. This
module is the production side of the same work: it fits the chosen models with
the hyperparameters that search settled on, saves them, and exposes one
`score_request` entry point for the agent to call.

Hyperparameters are fixed rather than re-searched. Re-running RandomizedSearchCV
inside the serving path would add minutes to every cold start to rediscover
values already known.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from syntetic_data_complete import (
    CATEGORICAL_FEATURES,
    DEPARTMENTS,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMNS,
    generate_procurement_ml_data,
)

RANDOM_STATE = 42
MODEL_DIR = Path("models")

DEPT_COMPUTE_NEED = {name: cfg["compute_need"] for name, cfg in DEPARTMENTS.items()}

ENGINEERED = [
    "price_premium_ratio", "budget_utilisation", "cap_overrun_ratio",
    "spec_gap", "is_overspec", "is_underspec",
    "price_per_ram_gb", "price_per_storage_gb",
    "tickets_per_travel_day", "wear_exposure",
]

NUMERIC_ALL = NUMERIC_FEATURES + ENGINEERED


def engineer_features(frame):
    """Derive ratio and spec-adequacy features. Reads inputs only, never a label."""
    out = frame.copy()

    out["price_premium_ratio"] = (
        (out.requested_unit_price - out.historical_avg_price) / out.historical_avg_price
    )
    out["budget_utilisation"] = out.total_amount / out.dept_budget_remaining
    out["cap_overrun_ratio"] = (
        (out.requested_unit_price - out.dept_policy_cap) / out.dept_policy_cap
    )

    # The signed distance between what the machine can do and what the role
    # needs. Positive is over-provisioned, negative is under -- this is the
    # number the assistant turns into "32 GB berlebihan".
    capability = out.compute_tier + 0.55 * np.log2(out.ram_gb.clip(lower=1) / 8.0)
    out["spec_gap"] = capability - out.department.map(DEPT_COMPUTE_NEED)
    out["is_overspec"] = (out.spec_gap > 0.75).astype(int)
    out["is_underspec"] = (out.spec_gap < -0.75).astype(int)

    out["price_per_ram_gb"] = out.requested_unit_price / out.ram_gb
    out["price_per_storage_gb"] = out.requested_unit_price / out.storage_gb

    out["tickets_per_travel_day"] = out.it_tickets_last_year / (out.travel_days_per_month + 1.0)
    out["wear_exposure"] = out.travel_days_per_month * (1.0 - out.build_quality)

    return out


def build_matrix(frame):
    """Assemble X, dropping every target so no model can see another's label."""
    banned = set(TARGET_COLUMNS) | set(ID_COLUMNS)
    cols = [c for c in CATEGORICAL_FEATURES + NUMERIC_ALL if c not in banned]

    leaked = banned.intersection(cols)
    if leaked:
        raise AssertionError(f"target/id column leaked into features: {leaked}")

    return frame[cols]


def _preprocessor(scale_numeric):
    """scale_numeric=True for linear models; a no-op for trees, which split on rank."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", StandardScaler() if scale_numeric else "passthrough", NUMERIC_ALL),
    ])


def _pipe(model, scale_numeric=False):
    return Pipeline([("prep", _preprocessor(scale_numeric)), ("model", model)])


def train_all(n_records=6000, out_dir=MODEL_DIR, verbose=True):
    """Fit all four models plus the serving defaults, and persist the bundle."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = engineer_features(generate_procurement_ml_data(n_records))
    X = build_matrix(df)
    report = {}

    # --- UC1 suitability: RandomForest won on ROC-AUC / PR-AUC.
    y = df.target_uc1_is_match
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    uc1 = _pipe(RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=4, max_features="log2",
        random_state=RANDOM_STATE, n_jobs=-1,
    )).fit(Xtr, ytr)
    report["uc1_pr_auc"] = average_precision_score(yte, uc1.predict_proba(Xte)[:, 1])

    # --- UC4 approval: XGBoost with the loss reweighted for the 30% minority.
    y = df.target_uc4_is_approved
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    uc4 = _pipe(XGBClassifier(
        n_estimators=350, max_depth=4, learning_rate=0.06, subsample=0.85,
        colsample_bytree=0.8, min_child_weight=3, reg_lambda=3.0,
        scale_pos_weight=spw, eval_metric="logloss", tree_method="hist",
        random_state=RANDOM_STATE,
    )).fit(Xtr, ytr)
    report["uc4_pr_auc"] = average_precision_score(yte, uc4.predict_proba(Xte)[:, 1])

    # Threshold picked on TRAIN only -- choosing it on the test split would be
    # a second, quieter form of leakage.
    tr_proba = uc4.predict_proba(Xtr)[:, 1]
    prec, rec, thr = precision_recall_curve(ytr, tr_proba)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    uc4_threshold = float(thr[np.argmax(f1[:-1])])
    report["uc4_threshold"] = uc4_threshold
    report["uc4_f1"] = f1_score(yte, (uc4.predict_proba(Xte)[:, 1] >= uc4_threshold).astype(int))

    # --- UC2 lifespan: ridge beat both ensembles; the generator's survival term
    # is linear in the features, so a linear model is correctly specified.
    y = df.target_uc2_months_to_failure
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    uc2 = _pipe(Ridge(alpha=10.0, random_state=RANDOM_STATE), scale_numeric=True).fit(Xtr, ytr)
    report["uc2_r2"] = r2_score(yte, uc2.predict(Xte))

    # --- UC3 operating cost, NOT total TCO. Total TCO is dominated by
    # requested_unit_price, which is an input -- predicting it scores R^2 ~ 0.99
    # by echoing a feature back.
    y = df.target_uc3_opex_idr
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    uc3 = _pipe(XGBRegressor(
        n_estimators=550, max_depth=6, learning_rate=0.06, subsample=0.85,
        colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
        tree_method="hist", random_state=RANDOM_STATE,
    )).fit(Xtr, ytr)
    report["uc3_r2"] = r2_score(yte, uc3.predict(Xte))

    # Median/mode of every input, so a partially-specified request can still be
    # scored -- with the caller told exactly which fields were invented.
    defaults = {
        c: (df[c].mode()[0] if c in CATEGORICAL_FEATURES else float(df[c].median()))
        for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES
    }

    bundle = {
        "uc1": uc1, "uc2": uc2, "uc3": uc3, "uc4": uc4,
        "uc4_threshold": uc4_threshold,
        "defaults": defaults,
        "report": report,
    }
    joblib.dump(bundle, out_dir / "procurement_models.joblib")

    if verbose:
        print(f"Saved bundle to {out_dir / 'procurement_models.joblib'}")
        for k, v in report.items():
            print(f"  {k:16} {v:.4f}")

    return bundle


def load_all(out_dir=MODEL_DIR):
    path = Path(out_dir) / "procurement_models.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run ml_models.train_all() first.")
    return joblib.load(path)


# Fields a caller must supply for the scores to mean anything. Everything else
# falls back to a median, which is fine for wear_exposure and misleading for a
# budget -- hence the split.
REQUIRED_FOR_SUITABILITY = ["department", "ram_gb", "compute_tier"]
REQUIRED_FOR_APPROVAL = ["requested_unit_price", "dept_budget_remaining", "dept_policy_cap"]

# A few fields have an obviously safe fallback that the training median is not.
# `quantity` is the one that matters: the median order is ~13 units, so an
# unspecified quantity silently turned a 30M laptop into a 390M request and
# drove every approval probability to near zero. One unit is what a person
# asking about "a laptop" means.
SAFE_DEFAULTS = {"quantity": 1}


def score_request(bundle, fields):
    """Score a partially-specified request; report what had to be assumed.

    `approval_*` is withheld when the budget context is missing rather than
    computed from a median budget, which would look authoritative and mean
    nothing.
    """
    defaults = bundle["defaults"]

    unknown = set(fields) - set(defaults)
    if unknown:
        raise KeyError(f"unknown field(s): {sorted(unknown)}")

    row = {**defaults, **SAFE_DEFAULTS, **{k: v for k, v in fields.items() if v is not None}}
    supplied = {k for k, v in fields.items() if v is not None}
    assumed = sorted(set(defaults) - supplied)

    # A quoted price with no historical reference is not a 36% premium over the
    # fleet median -- it is simply the price. Anchoring the two together keeps
    # price_premium_ratio at zero instead of inventing an overcharge.
    if "historical_avg_price" not in supplied and "requested_unit_price" in supplied:
        row["historical_avg_price"] = row["requested_unit_price"]

    row["total_amount"] = row["requested_unit_price"] * row["quantity"]

    frame = engineer_features(pd.DataFrame([row]))
    matrix = build_matrix(frame)

    result = {
        "suitability_proba": float(bundle["uc1"].predict_proba(matrix)[0, 1]),
        "spec_gap": float(frame.spec_gap.iloc[0]),
        "expected_lifespan_months": float(bundle["uc2"].predict(matrix)[0]),
        "expected_opex_idr": float(bundle["uc3"].predict(matrix)[0]),
        "assumed_fields": assumed,
    }
    result["expected_total_tco_idr"] = result["expected_opex_idr"] + row["total_amount"]

    if all(f in supplied for f in REQUIRED_FOR_APPROVAL):
        proba = float(bundle["uc4"].predict_proba(matrix)[0, 1])
        result["approval_proba"] = proba
        result["approval_decision"] = (
            "likely approved" if proba >= bundle["uc4_threshold"] else "likely rejected"
        )
    else:
        result["approval_proba"] = None
        result["approval_decision"] = "unknown"
        result["approval_blocked_on"] = [f for f in REQUIRED_FOR_APPROVAL if f not in supplied]

    return result


if __name__ == "__main__":
    train_all()
