"""
Governance / anti-corruption indicators computed from procurement_history.csv.

IMPORTANT — responsible terminology: nothing here labels a transaction as
"corruption." Every flag is a *potential indicator that a human reviewer
should look at*, phrased as such throughout the UI ("potential anomaly",
"requires review", "outside benchmark range"). See app.py copy.

Split-purchase heuristic (APPROVAL_THRESHOLD_RP):
  Flags pairs of purchase orders for the same business unit + same
  specification + same vendor, placed within a short window of each
  other, where EACH individual order is below the stated Director
  approval threshold (Rp200,000,000 — see assumptions.csv / the sample
  RAG question about approval thresholds). Two small orders that could
  have been one larger order, placed close together, is a classic
  textbook indicator worth a second look — it is not proof of anything
  improper on its own.
"""
import numpy as np
import pandas as pd

APPROVAL_THRESHOLD_RP = 200_000_000
SPLIT_WINDOW_DAYS = 14


def detect_split_purchases(history: pd.DataFrame, threshold: float = APPROVAL_THRESHOLD_RP,
                            window_days: int = SPLIT_WINDOW_DAYS) -> pd.DataFrame:
    flagged_pairs = []
    for (unit, spec, vendor), g in history.groupby(["business_unit", "spec_id", "vendor"]):
        g = g.sort_values("po_date")
        recs = g[["po_id", "po_date", "current_quote_total_rp"]].to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                gap = (recs[j]["po_date"] - recs[i]["po_date"]).days
                if gap > window_days:
                    break
                if recs[i]["current_quote_total_rp"] < threshold and recs[j]["current_quote_total_rp"] < threshold:
                    flagged_pairs.append(dict(
                        business_unit=unit, spec_id=spec, vendor=vendor,
                        po_id_1=recs[i]["po_id"], po_id_2=recs[j]["po_id"],
                        date_1=recs[i]["po_date"].date(), date_2=recs[j]["po_date"].date(),
                        gap_days=gap, value_1=recs[i]["current_quote_total_rp"], value_2=recs[j]["current_quote_total_rp"],
                        combined_value=recs[i]["current_quote_total_rp"] + recs[j]["current_quote_total_rp"],
                    ))
    return pd.DataFrame(flagged_pairs)


def evidence_coverage(history: pd.DataFrame) -> dict:
    n = len(history)
    if n == 0:
        return dict(pct_with_benchmark=0.0, pct_with_doc=0.0, n=0)
    return dict(
        pct_with_benchmark=round(history["has_benchmark_reference"].mean() * 100, 1),
        pct_with_doc=round(history["has_supporting_doc"].mean() * 100, 1),
        n=n,
    )


def outside_benchmark_summary(flagged: pd.DataFrame) -> dict:
    n = len(flagged)
    n_outside = int(flagged["outside_benchmark"].sum()) if "outside_benchmark" in flagged.columns else 0
    return dict(n_total=n, n_outside=n_outside, pct_outside=round(n_outside / n * 100, 1) if n else 0.0)


# ---------------------------------------------------------------------------
# Potential-corruption indicators — human-in-the-loop screening
# ---------------------------------------------------------------------------
# Built from the request feature set (commercial ratios, vendor signals,
# spec-fit, price density). Deliberately a TRANSPARENT ADDITIVE SCORE rather
# than a model output: this queue names individual requests, so a reviewer has
# to be able to see which indicators fired and check them. An opaque score
# cannot be acted on or appealed.
#
# HITL by design: the system only surfaces candidates for review. It does not
# determine that anything improper occurred — a human investigates and
# concludes. Nothing here is evidence.
#
# NOTE on units: dept_policy_cap is a PER-UNIT price ceiling (median ~Rp22jt),
# not an order ceiling. It must be compared against requested_unit_price, never
# against total_amount (median ~Rp274jt) -- that comparison fires on ~98% of
# rows and is meaningless.
# With 9 indicators, >=4 queues 872 requests (22%) -- too many to review.
# >=5 gives 251 (6.3%), a workable audit sample.
CORRUPTION_FLAG_THRESHOLD = 5

CORRUPTION_INDICATORS = {
    "price_over_cap": "Harga unit melebihi policy cap departemen",
    "price_hugs_cap": "Harga tepat di bawah cap (90–100%) — pola threshold gaming",
    "price_above_history": "Harga >15% di atas rata-rata riwayat mesin sejenis",
    "price_density_high": "Harga per GB jauh di atas median kelompok spek yang sama",
    "vendor_risk_high": "Skor risiko vendor tinggi (>0.7)",
    "vendor_not_official": "Vendor bukan rekanan resmi",
    "is_urgent": "Ditandai urgent — melewati jalur kontrol normal",
    "spec_not_matched": "Spesifikasi tidak cocok dengan kebutuhan peran",
    "budget_heavy": "Order menyerap >50% sisa anggaran departemen",
}


def corruption_flag_indicators(requests: pd.DataFrame) -> pd.DataFrame:
    """One row per request with each indicator as a boolean plus a total score.

    Returns the input frame with `flag_<name>` columns, `risk_score`, and
    `indicators_fired` (a readable summary for the reviewer)."""
    df = requests.copy()
    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)

    price, hist = num("requested_unit_price"), num("historical_avg_price")
    cap, total = num("dept_policy_cap"), num("total_amount")
    budget, risk = num("dept_budget_remaining"), num("vendor_risk_score")
    ram, storage = num("ram_gb"), num("storage_gb")

    # Price per GB of RAM+storage, compared within the same compute tier so a
    # workstation is not judged against an admin laptop.
    density = price / (ram.fillna(0) + storage.fillna(0) / 10).replace(0, np.nan)
    tier_median = density.groupby(df.get("compute_tier", pd.Series(1, index=df.index))).transform("median")

    df["flag_price_over_cap"] = (price > cap).fillna(False)
    df["flag_price_hugs_cap"] = ((price / cap >= 0.90) & (price / cap <= 1.00)).fillna(False)
    df["flag_price_above_history"] = ((price / hist - 1) > 0.15).fillna(False)
    df["flag_price_density_high"] = (density > tier_median * 1.25).fillna(False)
    df["flag_vendor_risk_high"] = (risk > 0.7).fillna(False)
    df["flag_vendor_not_official"] = (num("vendor_is_official").fillna(1) == 0)
    df["flag_is_urgent"] = (num("is_urgent").fillna(0) == 1)
    df["flag_spec_not_matched"] = (num("target_uc1_is_match").fillna(1) == 0)
    df["flag_budget_heavy"] = ((total / budget) > 0.50).fillna(False)

    cols = [f"flag_{k}" for k in CORRUPTION_INDICATORS]
    df["risk_score"] = df[cols].sum(axis=1).astype(int)
    df["indicators_fired"] = [
        " · ".join(CORRUPTION_INDICATORS[k] for k in CORRUPTION_INDICATORS if row[f"flag_{k}"])
        for _, row in df[cols].iterrows()
    ]
    return df


def corruption_review_queue(requests: pd.DataFrame,
                            threshold: int = CORRUPTION_FLAG_THRESHOLD) -> pd.DataFrame:
    """Requests meeting the indicator threshold, highest score first. These are
    candidates for human review — not findings."""
    scored = corruption_flag_indicators(requests)
    return scored[scored.risk_score >= threshold].sort_values("risk_score", ascending=False)
