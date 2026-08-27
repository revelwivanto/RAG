# Capstone 2.0 — Router + Four ML Models + Sourcing Agent

## Purpose

A three-lane procurement assistant where each turn is routed to whichever subsystem can answer it: retrieval for history, ML for scoring a named machine, agent for finding options.

---

## How It Works

### The Six Gates (Router)

Requests evaluated left-to-right; first match wins. Order is the design.

| Gate | Condition | Lane | Action | Example |
|---|---|---|---|---|
| **G0** | Spec not in catalogue | — | Correct and ask | "Apple does not ship Core i7" |
| **G1** | Question about documents | RAG | Retrieve | "Apa kebijakan budget?" |
| **G2** | One fully-specified candidate | ML | Score machine | "Beli MacBook M3 16GB Design" |
| **G3** | Goal + constraint, no candidate | Agent | Search + relax | "Cari laptop Design, 5 juta" |
| **G4** | Many requests | Batch | (stub) | "Review semua request" |

### G0: Catalogue Validation

Before any model runs, specs checked against hard catalogue. 

**Why:** "Core i7" encodes cleanly even though no MacBook ships with it. A model would score it and return confident nonsense.

### G1–G3: The Three Lanes

**Lane 1 (G1): RAG** 
- Cost: 1x tokens (retrieval is cheap)

**Lane 2 (G2): ML Scoring**
- Cost: 2x tokens (model inference is local)

**Lane 3 (G3): Sourcing Agent**
- Cost: 5-8x tokens (multiple search loops)

### Sourcing Agent Loop

`
Plan → Search → Score → Evaluate
               ↓
             No pass?
               ↓
            Relax ladder → Replan
`

**Relaxation order (always followed):**
1. Widen price band by 15%
2. Drop non-binding specs (screen, storage)
3. Admit refurbished and prior-gen units
4. Lower compute tier (not below role floor)

**Hard floor (never relaxed):**
- Exceeds departmental policy cap
- Under-provisioned (spec_gap < 0)
- Flagged as a suspected scam
- OS mismatch (Windows-locked dept, macOS candidate)

---

## Notebook Cells

| Cell | Task | Duration |
|---|---|---|
| 1–3 | Setup, load Kaggle Secrets | instant |
| 4 | Train four ML models (6K records) | 2 min |
| 5 | Embed 2.2K chunks → Qdrant | 3 min (first run) |
| 6 | Release embedding model VRAM | instant |
| 7 | Launch vLLM localhost:8000 | 5–10 min (first run) |
| 8–10 | Offline tests (no LLM) | 30 sec |
| 11 | Write app.py | instant |
| 12 | Launch Streamlit + ngrok | 30 sec |
| 13 | Shutdown | instant |

---

## Example Conversation

**Turn 1: User asks for a non-existent config**
`
USER: "Beli Macbook intel i7 32GB buat AMGR Design"

Extract: brand=Apple, cpu="intel i7", ram=32, dept=Design, role=Manager
Validate: FAIL
  → Apple does not ship Core i7 configurations
Gate: G0
Output: "Apple does not ship Core i7. Tersedia: M3, M3 Pro, M4"
`

**Turn 2: User specifies a valid candidate**
`
USER: "M3 Pro saja, 16GB, budget 36 juta"

Extract: brand=Apple, model=MacBook Pro 14, ram=16, price=36M
Validate: PASS
Gate: G2 (fully specified)
Models:
  UC1 = 0.68 (suitability)
  UC2 = 56 months (lifespan)
  UC3 = 3.2M IDR (opex)
  UC4 = 0.88 (approval probability)
Output: "Untuk AMGR Design, M3 Pro cocok (68% match). Tahan 56 bulan, 
         maintenance 3.2M. Approval odds tinggi (88%)."
`

**Turn 3: User asks a policy question**
`
USER: "Berapa budget biasanya untuk Design?"

Gate: G1 (question about documents)
Output: [retrieves from Qdrant] "Kebijakan budget Design adalah..."
`

---

## Key Architectural Choices

### 1. Request State in One Place

Slots accumulate across turns. Laptop arrives in turn 1, budget in turn 5. Without one home for the object, the router reads a half-built request and mis-gates it.

### 2. G0 Before the Router

Validation runs on raw request. Only if specs are plausible do they enter the router. Prevents garbage-in scenarios.

### 3. G2 Precedes G3 Deliberately

User who names a machine wants it *assessed*, not replaced. Routing to agent would find alternatives — a different act.

### 4. ML Output Is Four Numbers, Not One Verdict

Instead of one approve/reject, LLM gets:
- Suitability (does it fit the role?)
- Lifespan (how long does it last?)
- Opex (how much does it cost to operate?)
- Approval odds (does policy allow it?)

These can agree or conflict, letting LLM explain nuance.

### 5. Defaults Are Reported, Never Silent

Missing fields default to training median/mode. Response lists which were guessed. This becomes LLM's follow-up question ("Berapa budget?") rather than a silent assumption.

---

## Limitations

1. **Batch lane (G4) is a stub.** Not implemented yet.

2. **Synthetic data everywhere except graph_analysis.** Only graph_analysis.ipynb runs on real scraped Tokopedia listings. ML and routing train on generated procurement records.

3. **Small catalogue.** 12 laptop models in tests. Real deployment would need 1000+ configurations across brands.

4. **No marketplace APIs.** Sourcing agent searches the internal catalogue. Production version would query Tokopedia, Shopee, and other marketplaces for real listings with actual prices and seller reputation.

---

## Integration with Graph Analysis

Graph analysis (graph_analysis.ipynb) identifies seller families from scraped Tokopedia listings. Sourcing agent optionally accepts a seller_families dict and emits disclosures when presented options come from the same family:

`python
from graph_analysis import load_seller_families

families = load_seller_families()
agent = SourcingAgent(bundle, seller_families=families)
result = agent.run(slots)

# result["disclosures"] = [
#     "Agres ID Surabaya shares listing text with 3 other storefronts 
#      (Agres ID Electronics, Gateway Indonesia, LAPTOP SHOP ID). 
#      Quotes from these sellers are not independent."
# ]
`

---

## Next Steps (Not Yet Built)

1. **Marketplace sourcing** — Replace catalogue enumeration with live API queries
2. **G4 batch lane** — Process many requests at once, accumulate findings
3. **Feedback loop** — Log predictions vs outcomes (when historical data available), re-fit
4. **Policy engine** — Declarative constraints in G0 instead of hard-coded rules
5. **Multi-turn context** — Use conversation history to resolve ambiguities
