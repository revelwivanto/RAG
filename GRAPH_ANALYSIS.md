# Graph Analysis — Seller Network from Real Scraped Data

## Purpose

The **only genuinely real-world data** in this project: 120 laptop listings scraped from Tokopedia on 2026-08-27. Graph analysis audits the data and recovers seller families — storefronts that share listing text and are therefore part of the same operation.

This is not hypothetical. Real findings, real data. Real scam-detector bug, too.

---

## How It Works

### Two Problems, Two Solutions

#### Problem 1: The Scam Detector Is Wrong on Every Row It Fires On

**Status quo:** is_suspected_scam flag triggers when price < 3.5M IDR

**Reality:** All 26 flagged listings are accessories (keyboards, chargers, stands, a backpack, two promo bonuses) caught by a single rule that has no notion of *what* is being priced.

**Impact:** The flag is indexed into Qdrant and used as a hard floor in the sourcing agent. A 100% false-positive detector is worse than none — it looks like coverage.

**Data breakdown:**
- 120 total listings
- 39 accessories (keyboards, chargers, etc.)
- 81 genuine laptops
- 26 flagged as scam
  - 26 were accessories
  - 0 were actual laptops

**Action:** Retire the price-threshold rule. Replace with model-aware floor: only flag cheap items that are actually laptops.

---

#### Problem 2: Are Sellers Actually Independent?

**Observation:** Some storefronts have near-identical listing text across different machines. This might indicate:
1. A shared distributor feed
2. One operation running multiple storefronts
3. Copy-paste between competitors

**Method:** Build a graph where sellers are linked by near-duplicate titles (Jaccard similarity > 0.80 on listing text). Seller names never enter the comparison, so any clustering we recover is independent discovery.

**Result:** Five seller families, validated by the fact that the method recovered "Agres ID" storefronts without ever comparing names.

---

## The Three Graph Constructions

### Construction 1: Bipartite (Seller ← → Model)

Edge = "this seller lists this model"

**Verdict:** Centrality adds nothing.
- PageRank vs degree: Spearman ρ = 0.82
- Top-5 by PageRank = Top-5 by degree (only internal order differs)
- Conclusion: A seller's PageRank is just a smoothed degree count

**Lesson:** Running PageRank and reporting top nodes dressed in "influential entity" language is decoration. alue_counts() would have given the same answer.

---

### Construction 2: Co-listing Projection (Seller ← → Seller)

Edge = "we both list the same model"

**Verdict:** Weak signal; popularity, not relationship.
- 51 sellers, 237 edges, modularity 0.361
- Modularity < 0.3 = barely more structure than random graph
- Problem: 13 sellers listing a Vivobook 14 is not a community; Vivobook 14 is popular

**Lesson:** An edge that means "we both sell the same thing" in a market where most sellers stock the same few models is close to a complete graph carrying no information.

---

### Construction 3: Title Similarity Graph (Seller ← → Seller)

Edge = "our listing titles are near-duplicates"

**Verdict:** This one works.

**Why:**
1. **Transitive:** If A~B and B~C, then {A,B,C} is a cluster even when A and C never overlap directly. A groupby cannot produce this.
2. **Independent discovery:** Seller names are never compared, only listing text. Any same-business pair the method recovers is validation that the clustering is real.

**Parameters:**
- Threshold: 0.80 Jaccard similarity (chosen by sweep)
- Communities: Louvain algorithm
- Modularity: 0.428 (meaningful, not strong)

**Families recovered:**

| Family | Members | Listings | Locations | Price Gap |
|---|---|---|---|---|
| 1 | 5 sellers | 12 | Jakarta Pusat, Barat, Bogor | 23% median |
| 2 | 4 sellers | 21 | Jakarta Utara, Surabaya, Depok | 15% median |
| 3 | 4 sellers (Agres ID x3 + Gateway) | 16 | Jakarta Utara | 18% median |
| 4 | 2 sellers | 2 | Jakarta Utara, Tangerang | 0% median |
| 5 | 2 sellers | 3 | Jakarta Pusat, Selatan | 6% median |

**Validation:** The method, which never compares seller names, placed Agres ID Electronics and Agres ID Surabaya in the same family — and at looser thresholds pulled in Agres ID Bintaro, Jakarta Utara, and Tangerang. These are transparently one business running storefronts under one banner. **The graph found them from product text alone.**

---

## Price Dispersion: The Companion Finding

Separate from the graph: compare what different sellers charge for the same machine.

**Top spreads:**
- ROG STRIX G16: 29.3M – 81.0M IDR (64% spread, 2 sellers)
- TUF GAMING: 13.6M – 30.1M IDR (55% spread, 3 sellers)
- VIVOBOOK 14: 9.4M – 16.1M IDR (42% spread, 13 sellers)

**Key insight:** A 42% spread is suspicious *only if the quotes are independent.* If all 13 sellers listing a Vivobook are in one family, it's one supplier quoting 13 times, not 13 independent offers.

---

## How to Use These Results

### In the Sourcing Agent

The agent optionally accepts a seller_families dict:

`python
from graph_analysis import load_seller_families, family_warning

families = load_seller_families()  # → models/seller_families.json
agent = SourcingAgent(bundle, seller_families=families)

result = agent.run(slots)
# If a presented option has a seller in a family:
# result["disclosures"] = [
#     "Agres ID Surabaya shares listing text with 3 other storefronts 
#      (Agres ID Electronics, Gateway Indonesia, LAPTOP SHOP ID). 
#      Quotes from these sellers are not independent."
# ]
`

**This is a disclosure, not a floor.** A shared distributor feed is ordinary retail. The actionable statement is narrower: "these quotes are not independent."

### Data Quality Audit

Before running any graph algorithm, audit the data:

`python
from graph_analysis import audit_scam_flags

audit = audit_scam_flags(listings)
# → {
#     "total": 120,
#     "accessories": 39,
#     "laptops": 81,
#     "flagged": 26,
#     "flagged_that_are_accessories": 26,
#     "flagged_that_are_laptops": 0,
#     "reason_vocabulary": {"price_far_below_market(<3500000)": 26}
# }
`

---

## Code Flow

`
load_listings()
  ↓
audit_scam_flags()  ← Data quality check
  ↓
classify_listings()  ← Separate accessories from laptops
  ↓
build_bipartite()   ← Three graph constructions
build_colisting_projection()
build_family_graph()
  ↓
detect_families()   ← Louvain communities
  ↓
export_seller_families()  ← Persist seller → family mapping
  ↓
family_warning()  ← Format disclosure for LLM
`

---

## Limitations (Honest Accounting)

### What IS Established

1. The scam flag is wrong on every row it fires on (26/26 false positives)
2. Five seller families exist in the Asus/Tokopedia sample
3. The method recovered known storefronts (Agres ID) without comparing names
4. Quotes within a family are not independent

### What IS NOT Established

1. **Nothing about fraud.** A shared distributor feed is ordinary retail.
2. **Nothing about vendor concentration in BNI's procurement.** The procurement records have endor_risk_score and endor_is_official but no vendor *identity*. No division↔vendor edge exists to build; inventing one would only rediscover the generator's assumptions.
3. **Nothing beyond Asus on Tokopedia.** The scrape is one brand on one marketplace. This is the structure of *the Asus reseller network on Tokopedia*, not Indonesian laptop retail.

### What Would Raise the Ceiling

1. **Seller registration numbers or bank accounts.** Would turn "shared listing text" into genuine identity resolution.
2. **A second marketplace.** The SELLS_ON relation currently points at one constant (Tokopedia). With Shopee, the relation would carry information.
3. **Real procurement records with vendor names.** The division↔vendor graph — the most-requested analysis — would finally be something other than an echo of synthetic data.

### On Scale

- 51 sellers is small
- Louvain modularity 0.43 is meaningful but not strong
- Five families is a lead worth a human check, not a final answer
- The technique is sound and would sharpen considerably on a larger scrape

---

## How to Run

### Audit the data
`python
import graph_analysis as ga

listings = ga.load_listings()
audit = ga.audit_scam_flags(listings)
print(audit)
`

### Find seller families
`python
families = ga.export_seller_families(listings)
# → models/seller_families.json

for seller in ["Agres ID Surabaya", "Gateway Indonesia Comp"]:
    warning = ga.family_warning(families, seller)
    print(warning)
`

### Check centrality honestly
`python
B = ga.build_bipartite(listings)
check = ga.centrality_vs_degree(B, kind="seller")
print(f"PageRank vs degree correlation: {check['spearman_rho']:.3f}")
print(f"Same top-5? {check['same_top5']}")
`

---

## Integration Notes

Graph results feed into the sourcing agent as an optional parameter. The agent currently sources from the catalogue (which has no seller identity), so disclosures don't fire yet. They will fire once the agent sources from marketplace APIs and candidates carry seller names.

**Status:** Wired up, tested, inactive until marketplace integration.
