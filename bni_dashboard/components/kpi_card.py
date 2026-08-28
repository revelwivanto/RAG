"""Executive KPI cards with a small count-up animation. Pure presentation —
all numbers are passed in already computed from the data layer, nothing
is invented here."""
import streamlit.components.v1 as components
from components.styling import NAVY, ORANGE, TEXT_MUTED, CARD, BORDER, DATA_TYPE_COLORS

_BADGE_LABELS = {
    "observed": "Observed / Real", "synthetic": "Synthetic", "illustrative": "Illustrative",
    "assumption": "Assumption", "assumption-based": "Assumption-based",
    "calculated": "Calculated", "regulatory_fact": "Regulatory Fact",
}


def kpi_row(cards: list[dict], height: int = 168):
    """cards: [{label, value, prefix, suffix, decimals, data_type, note}]"""
    card_html = []
    for i, c in enumerate(cards):
        color = DATA_TYPE_COLORS.get(c.get("data_type", ""), TEXT_MUTED)
        badge_label = _BADGE_LABELS.get(c.get("data_type", ""), c.get("data_type", ""))
        note = c.get("note", "")
        card_html.append(f"""
        <div class="kpi-card">
          <div class="kpi-badge" style="background:{color}1a;color:{color};border:1px solid {color}55;">{badge_label}</div>
          <div class="kpi-value" data-target="{c['value']}" data-decimals="{c.get('decimals', 0)}"
               data-prefix="{c.get('prefix','')}" data-suffix="{c.get('suffix','')}" id="kpi-{i}">0</div>
          <div class="kpi-label">{c['label']}</div>
          {f'<div class="kpi-note">{note}</div>' if note else ''}
        </div>""")

    html = f"""
    <div class="kpi-row">{''.join(card_html)}</div>
    <style>
      .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; font-family:'Inter','Segoe UI',sans-serif; }}
      .kpi-card {{ flex:1; min-width:190px; background:{CARD}; border:1px solid {BORDER}; border-radius:14px;
                   padding:16px 18px; box-shadow:0 2px 10px rgba(0,39,77,0.06); position:relative; }}
      .kpi-badge {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:0.62rem;
                    font-weight:700; letter-spacing:0.03em; text-transform:uppercase; margin-bottom:10px; }}
      .kpi-value {{ font-size:1.9rem; font-weight:800; color:{NAVY}; line-height:1.1; }}
      .kpi-label {{ color:{TEXT_MUTED}; font-size:0.82rem; margin-top:4px; }}
      .kpi-note {{ color:{TEXT_MUTED}; font-size:0.72rem; margin-top:6px; border-top:1px dashed {BORDER}; padding-top:6px; }}
    </style>
    <script>
      const els = document.querySelectorAll('.kpi-value');
      els.forEach(el => {{
        const target = parseFloat(el.getAttribute('data-target'));
        const decimals = parseInt(el.getAttribute('data-decimals'));
        const prefix = el.getAttribute('data-prefix');
        const suffix = el.getAttribute('data-suffix');
        let cur = 0;
        const steps = 40;
        const inc = target / steps;
        let n = 0;
        const timer = setInterval(() => {{
          n += 1; cur += inc;
          if (n >= steps) {{ cur = target; clearInterval(timer); }}
          el.textContent = prefix + cur.toLocaleString('id-ID', {{minimumFractionDigits: decimals, maximumFractionDigits: decimals}}) + suffix;
        }}, 16);
      }});
    </script>
    """
    components.html(html, height=height, scrolling=False)
