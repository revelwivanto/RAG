from components.styling import NAVY, ORANGE, CARD, BORDER, TEXT_MUTED


def _flow(steps: list[str], accent_last: bool = False) -> str:
    nodes = []
    for i, s in enumerate(steps):
        is_last = accent_last and i == len(steps) - 1
        bg = f"{ORANGE}1a" if is_last else CARD
        border = ORANGE if is_last else BORDER
        color = ORANGE if is_last else NAVY
        nodes.append(f'<div class="flow-node" style="background:{bg};border-color:{border};color:{color};">{s}</div>')
        if i < len(steps) - 1:
            nodes.append(f'<div class="flow-arrow">&#8594;</div>')
    return f"""
    <div class="flow-row">{''.join(nodes)}</div>
    <style>
      .flow-row {{ display:flex; align-items:center; flex-wrap:wrap; gap:2px; font-family:'Inter','Segoe UI',sans-serif; }}
      .flow-node {{ border:1.5px solid {BORDER}; border-radius:10px; padding:10px 14px; font-size:0.82rem;
                    font-weight:600; text-align:center; box-shadow:0 1px 4px rgba(0,39,77,0.05); }}
      .flow-arrow {{ color:{TEXT_MUTED}; font-size:1.1rem; padding:0 6px; }}
    </style>
    """


def provenance_chain_html(chain: dict) -> str:
    steps = [
        f"Answer<br><span style='font-weight:400;font-size:0.72rem'>{chain['answer']}</span>",
        f"Citation<br><span style='font-weight:400;font-size:0.72rem'>{chain['citation']}</span>",
        f"Parent<br><span style='font-weight:400;font-size:0.72rem'>{chain['parent']}</span>",
        f"Document<br><span style='font-weight:400;font-size:0.72rem'>{chain['document']}</span>",
        f"{chain['page']}",
        f"Bounding box<br><span style='font-weight:400;font-size:0.72rem'>{chain['bbox']}</span>",
    ]
    return _flow(steps, accent_last=True)


def tax_normalization_flow_html(observed_label: str, ppn_label: str, comparable_label: str) -> str:
    return _flow([
        f"Marketplace observed price<br><span style='font-weight:400;font-size:0.72rem'>{observed_label}</span>",
        f"Tax normalization<br><span style='font-weight:400;font-size:0.72rem'>{ppn_label}</span>",
        f"Comparable benchmark price<br><span style='font-weight:400;font-size:0.72rem'>{comparable_label}</span>",
    ])
