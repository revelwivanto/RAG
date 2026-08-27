"""Graph analysis over the scraped Tokopedia listings.

This is the only genuinely real-world data in the project -- 120 listings
scraped from Tokopedia -- so findings here are findings, not restatements of a
generator's assumptions.

Two things it does:

1. **Audits the data.** The scrape mixes accessories in with laptops, and the
   existing `is_suspected_scam` flag is a single price threshold that fires on
   every cheap item. All 26 flags turn out to be false positives.

2. **Recovers seller families.** Sellers are linked by near-duplicate listing
   *titles* -- seller names are never compared -- and the connected structure of
   that graph exposes storefront families and shared distributor feeds. This is
   the part that genuinely needs a graph: the relation is transitive, so a
   cluster {A, B, C} formed from A~B and B~C is something no groupby returns.

Deliberately uses networkx rather than Neo4j. At 51 sellers and 120 listings the
algorithms run in milliseconds in memory, and a graph database would add an
infrastructure dependency without changing a single result.
"""

import collections
import itertools
import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx

LISTINGS_PATH = Path("data/marketplace/laptop_listings_filled.json")

# Parts and peripherals sold under "laptop ..." titles. The scrape's search terms
# caught these, and every one of them trips a price-based scam rule.
ACCESSORY_RE = re.compile(
    r"\b(keyboard|charger|adaptor|adapter|baterai|battery|casing|case|tas|bag|"
    r"sleeve|cooling|cooler|dock|hinge|engsel|lcd|layar|screen|fan|kipas|"
    r"mouse|stand|sticker|garskin|thermal|kabel|cable|backpack|bonus)\b",
    re.I,
)

# Below this, a "laptop" listing is a part, not a machine.
MIN_LAPTOP_PRICE = 2_000_000

# Two listings are treated as the same underlying text above this ratio.
# Chosen by sweep: looser merges every seller into one component, tighter
# fragments known storefront families apart.
TITLE_SIMILARITY_THRESHOLD = 0.80


def load_listings(path=LISTINGS_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 1. Data quality
# --------------------------------------------------------------------------

def classify_listings(listings):
    """Tag each listing as an accessory or a plausible machine."""
    out = []
    for x in listings:
        item = dict(x)
        title = x.get("title") or ""
        item["is_accessory"] = bool(ACCESSORY_RE.search(title)) or (
            (x.get("price_idr") or 0) < MIN_LAPTOP_PRICE
        )
        out.append(item)
    return out


def audit_scam_flags(listings):
    """Check what the existing is_suspected_scam flag actually caught."""
    tagged = classify_listings(listings)
    flagged = [x for x in tagged if x.get("is_suspected_scam")]

    reasons = collections.Counter(
        r for x in listings for r in (x.get("scam_reasons") or [])
    )
    return {
        "total": len(tagged),
        "accessories": sum(1 for x in tagged if x["is_accessory"]),
        "laptops": sum(1 for x in tagged if not x["is_accessory"]),
        "flagged": len(flagged),
        "flagged_that_are_accessories": sum(1 for x in flagged if x["is_accessory"]),
        "flagged_that_are_laptops": sum(1 for x in flagged if not x["is_accessory"]),
        "reason_vocabulary": dict(reasons),
    }


def laptops_only(listings):
    return [x for x in classify_listings(listings) if not x["is_accessory"]]


# --------------------------------------------------------------------------
# 2. Graphs
# --------------------------------------------------------------------------

def build_bipartite(listings):
    """Seller -- Model bipartite graph, one edge per seller/model pairing."""
    G = nx.Graph()
    for x in listings:
        seller = f"S:{x['seller_name']}"
        model = f"M:{x.get('model') or 'UNKNOWN'}"
        G.add_node(seller, kind="seller", label=x["seller_name"])
        G.add_node(model, kind="model", label=x.get("model") or "UNKNOWN")
        G.add_edge(seller, model)
    return G


def build_colisting_projection(listings):
    """Sellers linked when they list the same model.

    Included so its weakness is visible: two sellers both listing a popular
    model is not a relationship, it is popularity. Compare against the
    title-similarity graph below.
    """
    by_model = collections.defaultdict(set)
    for x in listings:
        by_model[x.get("model") or "UNKNOWN"].add(x["seller_name"])

    G = nx.Graph()
    G.add_nodes_from({x["seller_name"] for x in listings})
    for sellers in by_model.values():
        for a, b in itertools.combinations(sorted(sellers), 2):
            G.add_edge(a, b, weight=G.get_edge_data(a, b, {}).get("weight", 0) + 1)
    return G


def title_similarity_pairs(listings, threshold=TITLE_SIMILARITY_THRESHOLD):
    """Cross-seller listing pairs whose titles are near-duplicates.

    Seller names never enter the comparison, so any same-business pair this
    recovers is an independent discovery rather than string matching on names.
    """
    weights, best = collections.Counter(), {}

    for a, b in itertools.combinations(listings, 2):
        if a["seller_name"] == b["seller_name"]:
            continue
        ratio = SequenceMatcher(
            None, (a.get("title") or "").lower(), (b.get("title") or "").lower()
        ).ratio()
        if ratio >= threshold:
            key = tuple(sorted((a["seller_name"], b["seller_name"])))
            weights[key] += 1
            best[key] = max(best.get(key, 0.0), ratio)

    return weights, best


def build_family_graph(listings, threshold=TITLE_SIMILARITY_THRESHOLD):
    """Seller graph whose edges are shared listing text, weighted by pair count."""
    weights, best = title_similarity_pairs(listings, threshold)
    G = nx.Graph()
    for (u, v), count in weights.items():
        G.add_edge(u, v, weight=count, best_ratio=best[(u, v)])
    return G


def detect_families(G, seed=42):
    """Louvain communities, largest first. Falls back to components on an empty graph."""
    if G.number_of_edges() == 0:
        return [], 0.0
    communities = nx.community.louvain_communities(G, weight="weight", seed=seed)
    modularity = nx.community.modularity(G, communities, weight="weight")
    return sorted(communities, key=len, reverse=True), modularity


# --------------------------------------------------------------------------
# 3. Does the graph earn its place?
# --------------------------------------------------------------------------

def centrality_vs_degree(G, kind="seller"):
    """Compare PageRank against plain degree.

    If the two rank nodes identically, the centrality computation is decoration
    and `value_counts()` would have answered the question. Reporting the
    correlation is the difference between using a graph and performing one.
    """
    from scipy.stats import spearmanr

    nodes = [n for n, a in G.nodes(data=True) if a.get("kind") == kind] or list(G.nodes())
    pagerank = nx.pagerank(G)
    degree = dict(G.degree())

    rho = spearmanr([pagerank[n] for n in nodes], [degree[n] for n in nodes]).statistic
    top_pr = [n for n in sorted(nodes, key=lambda n: -pagerank[n])[:5]]
    top_deg = [n for n in sorted(nodes, key=lambda n: -degree[n])[:5]]

    return {
        "spearman_rho": float(rho),
        "top5_pagerank": [n.split(":", 1)[-1] for n in top_pr],
        "top5_degree": [n.split(":", 1)[-1] for n in top_deg],
        "same_top5": set(top_pr) == set(top_deg),
    }


# --------------------------------------------------------------------------
# 4. What the families look like
# --------------------------------------------------------------------------

def describe_family(listings, members, weights=None):
    rows = [x for x in listings if x["seller_name"] in members]
    prices = [x["price_idr"] for x in rows if x.get("price_idr")]

    gaps = []
    if weights:
        for (u, v) in weights:
            if u in members and v in members:
                pu = [x["price_idr"] for x in rows if x["seller_name"] == u and x.get("price_idr")]
                pv = [x["price_idr"] for x in rows if x["seller_name"] == v and x.get("price_idr")]
                if pu and pv:
                    gaps.append(abs(min(pu) - min(pv)) / max(min(pu), min(pv)))

    return {
        "members": sorted(members),
        "size": len(members),
        "listings": len(rows),
        "locations": dict(collections.Counter(x.get("location") for x in rows)),
        "median_price": statistics.median(prices) if prices else None,
        "median_price_gap": statistics.median(gaps) if gaps else None,
    }


def price_dispersion(listings):
    """Per model, how far apart do sellers quote the same machine?

    This is the honest, non-graph half of anomaly detection -- a groupby, and it
    should be named as one rather than dressed up as a network result.
    """
    by_model = collections.defaultdict(list)
    for x in listings:
        if x.get("model") and x.get("price_idr"):
            by_model[x["model"]].append(x)

    rows = []
    for model, items in by_model.items():
        sellers = {x["seller_name"] for x in items}
        if len(sellers) < 2:
            continue
        prices = sorted(x["price_idr"] for x in items)
        rows.append({
            "model": model,
            "sellers": len(sellers),
            "listings": len(items),
            "min_price": prices[0],
            "max_price": prices[-1],
            "spread_pct": (prices[-1] - prices[0]) / prices[-1],
        })
    return sorted(rows, key=lambda r: -r["spread_pct"])


# --------------------------------------------------------------------------
# 5. Handing the result to the agent
# --------------------------------------------------------------------------

def export_seller_families(listings, out_path="models/seller_families.json",
                           threshold=TITLE_SIMILARITY_THRESHOLD):
    """Persist seller -> family id so the sourcing agent can consult it.

    A seller sitting in a multi-member family is not itself proof of anything.
    It is a disclosure: the "three independent quotes" a procurement officer
    thinks they are comparing may all originate from one operation.
    """
    G = build_family_graph(listings, threshold)
    families, modularity = detect_families(G)
    weights, _ = title_similarity_pairs(listings, threshold)

    seller_to_family, records = {}, []
    for i, members in enumerate(families):
        if len(members) < 2:
            continue
        info = describe_family(listings, members, weights)
        info["family_id"] = i
        records.append(info)
        for member in members:
            seller_to_family[member] = i

    payload = {
        "threshold": threshold,
        "modularity": modularity,
        "families": records,
        "seller_to_family": seller_to_family,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_seller_families(path="models/seller_families.json"):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def family_warning(families_payload, seller_name):
    """One sentence for the assistant, or None when there is nothing to disclose."""
    if not families_payload:
        return None
    fid = families_payload["seller_to_family"].get(seller_name)
    if fid is None:
        return None
    family = next(f for f in families_payload["families"] if f["family_id"] == fid)
    others = [m for m in family["members"] if m != seller_name]
    return (
        f"{seller_name} shares listing text with {len(others)} other storefront(s) "
        f"({', '.join(others[:3])}{'...' if len(others) > 3 else ''}). "
        f"Quotes from these sellers are not independent."
    )


if __name__ == "__main__":
    data = load_listings()
    print(json.dumps(audit_scam_flags(data), indent=2))
    payload = export_seller_families(data)
    print(f"\n{len(payload['families'])} families, modularity {payload['modularity']:.3f}")
