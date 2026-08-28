"""
BNI-inspired visual identity. Colors are approximate / inspired by BNI's
public brand (deep navy + orange accent) — not verified against an
official brand guideline, and this dashboard is an internal prototype,
not an official BNI publication. Swap the hex values below if your
brand team provides exact specs.
"""
import streamlit as st

NAVY = "#00274D"
NAVY_LIGHT = "#0B3D66"
ORANGE = "#F37021"
ORANGE_DARK = "#D9531E"
BG = "#F7F8FA"
CARD = "#FFFFFF"
TEXT = "#1A2733"
TEXT_MUTED = "#64748B"
BORDER = "#E4E9F0"
GREEN = "#1E8A5F"
AMBER = "#B8860B"
RED = "#C0392B"

DATA_TYPE_COLORS = {
    "observed": GREEN,
    "synthetic": TEXT_MUTED,
    "illustrative": ORANGE_DARK,
    "assumption": AMBER,
    "assumption-based": AMBER,
    "internal_assumption": AMBER,
    "regulatory_fact": NAVY,
    "no_regulatory_basis": RED,
    "calculated": NAVY_LIGHT,
}

CATEGORICAL_SEQUENCE = [NAVY, ORANGE, "#4C82A6", "#F4A65A", "#7594B3", "#B8860B"]


def inject_global_css():
    st.markdown(f"""
    <style>
    .stApp {{ background: {BG}; }}
    html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }}
    h1, h2, h3 {{ color: {NAVY}; font-weight: 700; }}
    h4, h5 {{ color: {NAVY}; font-weight: 600; }}
    [data-testid="stSidebar"] {{ background: {NAVY}; }}
    [data-testid="stSidebar"] * {{ color: #EAF1F8 !important; }}
    [data-testid="stSidebar"] .stRadio label {{ font-size: 0.95rem; }}
    [data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 700; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; }}
    div[data-testid="stExpander"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; }}
    .block-container {{ padding-top: 1.6rem; }}
    hr {{ border-color: {BORDER}; }}
    .bni-header {{
        background: linear-gradient(120deg, {NAVY} 0%, {NAVY_LIGHT} 100%);
        padding: 28px 32px; border-radius: 14px; color: white; margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(0,39,77,0.18);
    }}
    .bni-header .accent {{ color: {ORANGE}; }}
    .bni-header .subtitle {{ color: #C9D9E8; font-size: 0.95rem; margin-top: 4px; }}
    .bni-badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.72rem;
                  font-weight:700; letter-spacing:0.02em; text-transform:uppercase; }}
    .section-tag {{ color:{ORANGE}; font-weight:700; letter-spacing:0.08em; font-size:0.78rem;
                     text-transform:uppercase; margin-bottom:2px; }}
    </style>
    """, unsafe_allow_html=True)


def data_badge(data_type: str) -> str:
    labels = {
        "observed": "Observed / Real", "synthetic": "Synthetic", "illustrative": "Illustrative",
        "assumption": "Assumption", "assumption-based": "Assumption-based",
        "internal_assumption": "Internal Assumption", "regulatory_fact": "Regulatory Fact",
        "no_regulatory_basis": "No Regulatory Basis Found", "calculated": "Calculated",
    }
    color = DATA_TYPE_COLORS.get(data_type, TEXT_MUTED)
    label = labels.get(data_type, data_type)
    return f'<span class="bni-badge" style="background:{color}20;color:{color};border:1px solid {color}55;">{label}</span>'
