"""Central place every page/component loads data from. Swap a CSV in data/
for a real one with the same columns and nothing else needs to change —
see README.md > Data Replacement Guide."""
from pathlib import Path
import csv
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def _read_marketplace_csv(path: Path) -> pd.DataFrame:
    """Read both the dashboard schema and the raw enriched listing export."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader([handle.readline()]))
        rows = []
        for line in handle:
            row = next(csv.reader([line]), [])
            if len(row) == len(header):
                rows.append(row)
                continue
            if line.startswith('"'):
                repaired = next(csv.reader([line[1:]]), [])
            else:
                repaired = row
            if len(repaired) > len(header):
                # The raw export sometimes leaves commas in title unquoted.
                repaired = repaired[:3] + [",".join(repaired[3:-19])] + repaired[-19:]
            if len(repaired) == len(header):
                rows.append(repaired)
        return pd.DataFrame(rows, columns=header)


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def _cpu_tier(cpu: pd.Series, title: pd.Series) -> pd.Series:
    text = cpu.fillna("").astype(str).str.upper()
    text = text.where(text.ne(""), title.fillna("").astype(str).str.upper())
    return pd.Series(
        pd.NA,
        index=text.index,
        dtype="string",
    ).mask(text.str.contains(r"I9|RYZEN 9|ULTRA 9|AI 9", regex=True), "premium").mask(
        text.str.contains(r"I7|RYZEN 7|ULTRA 7|CORE 5 [12]10H|RYZEN AI", regex=True), "high"
    ).mask(
        text.str.contains(r"I5|RYZEN 5|ULTRA 5|CORE 5", regex=True), "mid"
    ).mask(
        text.str.contains(r"I3|RYZEN 3|CELERON|N[345]0|ATHLON|PENTIUM", regex=True), "entry"
    ).fillna("mid")


def _normalize_raw_marketplace(df: pd.DataFrame) -> pd.DataFrame:
    """Map laptop_listings_all_cleaned_web_enriched_deduplicated.csv to app schema."""
    raw = df.copy()
    raw["source"] = raw["source"].astype(str).str.lower().replace("nan", "")
    raw["title"] = raw["title"].fillna("").astype(str)
    raw["cpu"] = raw["cpu"].fillna("").astype(str)
    raw["gpu"] = raw["gpu"].fillna("").astype(str)
    raw["price_idr"] = pd.to_numeric(raw["price_idr"], errors="coerce")
    raw["ram_gb"] = pd.to_numeric(raw["ram_gb"], errors="coerce")
    raw["storage_gb"] = pd.to_numeric(raw["storage_gb"], errors="coerce")
    raw["seller_rating"] = pd.to_numeric(raw["seller_rating"], errors="coerce")
    raw["sold_count"] = pd.to_numeric(raw["sold_count"], errors="coerce")
    raw["screen_size_in"] = pd.to_numeric(raw["screen_size_in"], errors="coerce")

    normalized = pd.DataFrame(index=raw.index)
    source_ids = raw["source_id"].where(raw["source_id"].notna(), pd.Series(raw.index, index=raw.index))
    normalized["listing_id"] = raw["source"].replace("", "unknown") + "-" + source_ids.astype(str)
    normalized["title"] = raw["title"]
    normalized["brand"] = raw["brand"].fillna("Unknown")
    normalized["cpu_brand"] = raw["cpu"].str.contains(r"RYZEN|AMD", case=False, regex=True).map({True: "AMD", False: "Intel"})
    normalized["cpu_tier"] = _cpu_tier(raw["cpu"], raw["title"])
    normalized["cpu_model"] = raw["cpu"].replace("", pd.NA)
    normalized["ram_gb"] = raw["ram_gb"]
    normalized["storage_gb"] = raw["storage_gb"]
    normalized["dedicated_gpu"] = raw["gpu"].str.strip().ne("")
    normalized["gpu_model"] = raw["gpu"].replace("", pd.NA)
    normalized["screen_in"] = raw["screen_size_in"]
    normalized["form_factor"] = raw["title"].str.contains(r"FLIP|2[- ]?IN[- ]?1|TOUCH", case=False, regex=True).map({True: "2-in-1 convertible", False: "clamshell"})
    normalized["condition"] = raw["condition"].fillna("new")
    normalized["price_rp"] = raw["price_idr"]
    normalized["orig_price_rp"] = pd.to_numeric(raw["original_price_idr"], errors="coerce")
    normalized["rating"] = raw["seller_rating"]
    normalized["sold_min"] = raw["sold_count"].fillna(0)
    normalized["marketplace"] = raw["source"].replace("", "Unknown").str.title()
    normalized["seller"] = raw["seller_name"].fillna("Unknown")
    normalized["seller_location"] = raw["location"].fillna("Unknown")
    normalized["source_url"] = raw["url"]
    normalized["checked_date"] = pd.to_datetime(raw["scraped_at"], errors="coerce").dt.date.astype("string")
    normalized["is_duplicate_listing"] = normalized.duplicated(subset=["title", "price_rp", "seller"], keep="first")
    normalized["is_bundle"] = normalized["title"].str.contains(r"BONUS|BUNDLE|PAKET", case=False, regex=True)
    normalized["is_promo"] = normalized["title"].str.contains(r"SALE|PROMO|FLASH|DISKON|PICKUP", case=False, regex=True)
    normalized["missing_spec"] = normalized[["cpu_model", "ram_gb", "storage_gb", "price_rp"]].isna().any(axis=1)
    normalized["is_price_outlier"] = False
    normalized["is_suspicious"] = _as_bool(raw["is_suspected_scam"])
    normalized["is_price_outlier"] = normalized["is_suspicious"]
    normalized["listing_status"] = "valid"
    normalized.loc[normalized["is_duplicate_listing"], "listing_status"] = "removed_duplicate"
    normalized.loc[normalized["is_suspicious"], "listing_status"] = "flagged_suspicious"
    normalized.loc[normalized["missing_spec"], "listing_status"] = "incomplete"
    normalized["data_type"] = "observed"
    return normalized.reset_index(drop=True)


@st.cache_data
def load_marketplace_listings() -> pd.DataFrame:
    path = DATA_DIR / "marketplace_listings.csv"
    df = _read_marketplace_csv(path)
    if {"price_idr", "seller_name", "is_suspected_scam"}.issubset(df.columns):
        df = _normalize_raw_marketplace(df)
    else:
        numeric_cols = ["ram_gb", "storage_gb", "screen_in", "price_rp", "orig_price_rp", "rating", "sold_min"]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        bool_cols = ["dedicated_gpu", "is_duplicate_listing", "is_bundle", "is_promo",
                     "missing_spec", "is_price_outlier", "is_suspicious"]
        for c in bool_cols:
            df[c] = _as_bool(df[c])
    return df


@st.cache_data
def load_procurement_history() -> pd.DataFrame:
    df = load_csv("procurement_history.csv")
    df["po_date"] = pd.to_datetime(df["po_date"])
    df["dedicated_gpu"] = df["dedicated_gpu"].astype(bool)
    df["has_benchmark_reference"] = df["has_benchmark_reference"].astype(bool)
    df["has_supporting_doc"] = df["has_supporting_doc"].astype(bool)
    return df


@st.cache_data
def load_procurement_catalog() -> pd.DataFrame:
    df = load_csv("procurement_laptops.csv")
    df["dedicated_gpu"] = df["dedicated_gpu"].astype(bool)
    return df


@st.cache_data
def load_scenarios() -> pd.DataFrame:
    scenarios = load_csv("procurement_scenarios.csv")
    scenarios["qty_tier"] = scenarios["qty_tier"].astype(int)
    scenarios = scenarios.set_index("qty_tier").sort_index()

    # Interpolate the assumption tiers so the calculator can show every unit quantity.
    quantities = sorted(set(range(1, 201)) | set(scenarios.index))
    expanded = scenarios.reindex(quantities)
    expanded["assumed_discount_pct"] = expanded["assumed_discount_pct"].interpolate(method="index").ffill()
    expanded["rationale"] = expanded["rationale"].fillna("Interpolated between documented discount tiers")
    expanded["source"] = expanded["source"].fillna("calculated from procurement_scenarios.csv")
    expanded["data_type"] = expanded["data_type"].fillna("calculated")
    return expanded.reset_index()


@st.cache_data
def load_assumptions() -> pd.DataFrame:
    return load_csv("assumptions.csv")


@st.cache_data
def load_legal_documents() -> pd.DataFrame:
    return load_csv("legal_documents.csv")


@st.cache_data
def load_rag_answers() -> pd.DataFrame:
    df = load_csv("rag_answers.csv")
    df["grounded"] = df["grounded"].astype(bool)
    df["has_citation"] = df["has_citation"].astype(bool)
    return df


@st.cache_data
def load_citations() -> pd.DataFrame:
    return load_csv("citations.csv")


@st.cache_data
def load_pilot_metrics() -> pd.DataFrame:
    return load_csv("pilot_metrics.csv")


@st.cache_data
def load_business_unit_impact() -> pd.DataFrame:
    return load_csv("business_unit_impact.csv")


@st.cache_data
def load_process_time() -> pd.DataFrame:
    return load_csv("process_time_comparison.csv")


@st.cache_data
def load_doc_prep_time() -> pd.DataFrame:
    return load_csv("doc_prep_time.csv")


@st.cache_data
def load_citation_validity() -> pd.DataFrame:
    return load_csv("citation_validity_by_category.csv")


@st.cache_data
def load_pilot_ramp() -> pd.DataFrame:
    return load_csv("pilot_ramp_curve.csv")


@st.cache_data
def load_roi_scenarios() -> pd.DataFrame:
    return load_csv("roi_scenarios.csv")


@st.cache_data
def load_citation_evaluations() -> pd.DataFrame:
    return load_csv("citation_evaluations.csv")


@st.cache_data
def load_investment_breakdown() -> pd.DataFrame:
    return load_csv("investment_breakdown.csv")


@st.cache_data
def load_process_annual_volume() -> pd.DataFrame:
    return load_csv("process_annual_volume.csv")
