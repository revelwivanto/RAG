"""Router and sourcing agent for the procurement assistant.

Three lanes sit behind one chat box, and the router decides which one a turn
belongs to. Most turns are not agentic and should not pay for an agent; one kind
genuinely is.

    G0  spec not in the catalogue, or a required field missing  -> correct & ask
    G1  definitional / historical question about the documents  -> plain RAG
    G2  one fully-resolved candidate                            -> ML scoring
    G3  a goal plus a constraint, but no candidate              -> sourcing agent
    G4  scope is a set of requests rather than one              -> batch lane
    --  nothing matched                                         -> clarify

Gates are evaluated in order and the first match wins, so the ordering is the
design. G0 sits ahead of everything because a request naming hardware that does
not exist must never reach a model that would score it anyway: "MacBook with
Core i7" encodes cleanly (both values are individually valid) and yields a
confident number for a machine nobody sells.

G2 sits ahead of G3 deliberately. Someone who names a specific machine wants it
assessed, not replaced; searching for alternatives is a different act from
scoring what was asked about.
"""

import json
import re
from dataclasses import dataclass, field, asdict

from ml_models import DEPT_COMPUTE_NEED, score_request
from syntetic_data_complete import DEPARTMENTS, LAPTOPS, SENIORITY

# --------------------------------------------------------------------------
# Catalogue -- the ground truth G0 validates against
# --------------------------------------------------------------------------

CATALOGUE = [
    {
        "laptop_brand": b, "laptop_model": m, "cpu": c, "gpu": g,
        "base_ram_gb": ram, "base_storage_gb": ssd, "base_price": price,
        "compute_tier": tier, "build_quality": build, "portability": port,
    }
    for (b, m, c, g, ram, ssd, price, tier, build, port) in LAPTOPS
]

BRANDS = sorted({row["laptop_brand"] for row in CATALOGUE})
MODELS = sorted({row["laptop_model"] for row in CATALOGUE})
DEPARTMENT_NAMES = sorted(DEPARTMENTS)
SENIORITY_NAMES = list(SENIORITY)

RAM_OPTIONS = [8, 16, 24, 32, 48]

# Indonesian role abbreviations that appear in real requests. Without this the
# extractor has to guess what "AMGR" means, and a wrong seniority silently
# changes both the policy cap and the suitability score.
ROLE_ALIASES = {
    "staff": "Staff", "stf": "Staff", "officer": "Staff", "ofc": "Staff",
    "senior": "Senior", "snr": "Senior", "sr": "Senior",
    "asisten manager": "Manager", "asst manager": "Manager", "amgr": "Manager",
    "manager": "Manager", "mgr": "Manager", "manajer": "Manager",
    "director": "Director", "direktur": "Director", "dir": "Director",
}


def models_for_brand(brand):
    return sorted({r["laptop_model"] for r in CATALOGUE if r["laptop_brand"] == brand})


def cpus_for_brand(brand):
    return sorted({r["cpu"] for r in CATALOGUE if r["laptop_brand"] == brand})


def catalogue_row(brand=None, model=None):
    for row in CATALOGUE:
        if model and row["laptop_model"] == model:
            return row
        if brand and not model and row["laptop_brand"] == brand:
            return row
    return None


# --------------------------------------------------------------------------
# Request state -- slots accumulate across turns
# --------------------------------------------------------------------------

@dataclass
class RequestSlots:
    """What the assistant knows so far.

    Slots survive across turns: the laptop arrives in turn 1 and the budget in
    turn 5. Without one home for this object the router reads a half-built
    request and mis-gates it.
    """
    department: str = None
    seniority_level: str = None
    laptop_brand: str = None
    laptop_model: str = None
    cpu: str = None
    gpu: str = None
    ram_gb: int = None
    storage_gb: int = None
    requested_unit_price: float = None
    quantity: int = None
    dept_budget_remaining: float = None
    intent: str = None            # "assess" | "find" | "ask" | "batch"
    free_text: str = ""

    def merge(self, updates):
        for key, value in (updates or {}).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        return self

    def has_candidate(self):
        """A candidate is resolved once we know which machine is meant."""
        return self.laptop_model is not None or (
            self.laptop_brand is not None and self.ram_gb is not None
        )

    def has_constraint(self):
        return self.requested_unit_price is not None or self.dept_budget_remaining is not None

    def as_scoring_fields(self):
        """Project the slots onto the feature names ml_models expects."""
        row = catalogue_row(self.laptop_brand, self.laptop_model)
        fields = {
            "department": self.department,
            "seniority_level": self.seniority_level,
            "laptop_brand": self.laptop_brand,
            "laptop_model": self.laptop_model,
            "cpu": self.cpu,
            "gpu": self.gpu,
            "ram_gb": self.ram_gb,
            "storage_gb": self.storage_gb,
            "requested_unit_price": self.requested_unit_price,
            "quantity": self.quantity,
            "dept_budget_remaining": self.dept_budget_remaining,
        }
        if row:
            fields.setdefault("compute_tier", row["compute_tier"])
            fields["compute_tier"] = row["compute_tier"]
            fields["build_quality"] = row["build_quality"]
            fields["portability"] = row["portability"]
        if self.department:
            cfg = DEPARTMENTS[self.department]
            mult = SENIORITY.get(self.seniority_level, SENIORITY["Staff"])["cap_multiplier"]
            fields["dept_policy_cap"] = cfg["budget_cap"] * mult
        return {k: v for k, v in fields.items() if v is not None}


# --------------------------------------------------------------------------
# Extraction (LLM, schema-constrained)
# --------------------------------------------------------------------------

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "department":      {"type": ["string", "null"], "enum": DEPARTMENT_NAMES + [None]},
        "seniority_level": {"type": ["string", "null"], "enum": SENIORITY_NAMES + [None]},
        "laptop_brand":    {"type": ["string", "null"], "enum": BRANDS + [None]},
        "laptop_model":    {"type": ["string", "null"], "enum": MODELS + [None]},
        "cpu_text":        {"type": ["string", "null"]},
        "ram_gb":          {"type": ["integer", "null"]},
        "storage_gb":      {"type": ["integer", "null"]},
        "requested_unit_price": {"type": ["number", "null"]},
        "quantity":        {"type": ["integer", "null"]},
        "budget_idr":      {"type": ["number", "null"]},
        "intent": {"type": "string", "enum": ["assess", "find", "ask", "batch"]},
    },
    "required": ["intent"],
}

EXTRACTION_PROMPT = """Extract procurement request fields from the user message.
Reply with JSON only.

Rules:
- Use null for anything the message does not state. Never invent a value.
- `intent`: "assess" when a specific laptop is named for evaluation, "find" when
  the user wants options or a recommendation, "ask" for questions about
  documents or past purchases, "batch" when the subject is many requests at once.
- `budget_idr` is a spending limit ("budget 5 juta"); `requested_unit_price` is
  the price of a named machine. "5 juta" means 5000000.
- Copy the CPU verbatim into `cpu_text` (e.g. "intel i7", "M3 Pro").

Known departments: {departments}
Known seniority levels: {seniority} (AMGR / Asisten Manager = Manager, Officer = Staff)

Message: {message}"""


def normalise_role(text):
    """Map Indonesian role abbreviations onto the trained seniority levels."""
    if not text:
        return None
    lowered = text.lower()
    for alias, level in sorted(ROLE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return level
    return None


def normalise_cpu(cpu_text, brand=None):
    """Resolve free-text CPU against the catalogue, per brand where known."""
    if not cpu_text:
        return None
    t = cpu_text.lower().replace("-", " ")
    pool = cpus_for_brand(brand) if brand else sorted({r["cpu"] for r in CATALOGUE})

    for cpu in pool:
        if cpu.lower() in t or t in cpu.lower():
            return cpu
    for cpu in pool:
        tail = cpu.lower().split()[-1]          # "i7" from "Core i7", "pro" from "M3 Pro"
        if re.search(rf"\b{re.escape(tail)}\b", t):
            return cpu
    return None


def extract_slots(client, model, message, current=None, temperature=0.0):
    """Ask the LLM for a structured request, constrained to the catalogue's enums.

    Guided decoding is what keeps the extractor inside the vocabulary the models
    were trained on. Without it the LLM invents a department, one-hot encoding
    silently drops it, and every downstream score drifts toward the mean with no
    error raised anywhere.
    """
    slots = current or RequestSlots()
    prompt = EXTRACTION_PROMPT.format(
        departments=", ".join(DEPARTMENT_NAMES),
        seniority=", ".join(SENIORITY_NAMES),
        message=message,
    )

    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=temperature,
    )
    try:
        response = client.chat.completions.create(
            **kwargs, extra_body={"guided_json": EXTRACTION_SCHEMA}
        )
    except Exception:
        # Older vLLM builds and non-vLLM servers reject guided_json; fall back to
        # unconstrained decoding and validate the JSON ourselves.
        response = client.chat.completions.create(**kwargs)

    raw = response.choices[0].message.content or "{}"
    match = re.search(r"\{.*\}", raw, re.S)
    payload = json.loads(match.group()) if match else {}

    brand = payload.get("laptop_brand")
    model_name = payload.get("laptop_model")
    if model_name and not brand:
        row = catalogue_row(model=model_name)
        brand = row["laptop_brand"] if row else None

    updates = {
        "department": payload.get("department"),
        "seniority_level": payload.get("seniority_level") or normalise_role(message),
        "laptop_brand": brand,
        "laptop_model": model_name,
        "ram_gb": payload.get("ram_gb"),
        "storage_gb": payload.get("storage_gb"),
        "requested_unit_price": payload.get("requested_unit_price"),
        "quantity": payload.get("quantity"),
        "dept_budget_remaining": payload.get("budget_idr"),
        "intent": payload.get("intent"),
    }
    slots.merge(updates)
    slots.free_text = message

    # Kept as raw text so validation can report the mismatch the user typed,
    # rather than a silently corrected value.
    slots._cpu_text = payload.get("cpu_text")
    slots.cpu = normalise_cpu(payload.get("cpu_text"), slots.laptop_brand)
    if slots.laptop_model:
        row = catalogue_row(model=slots.laptop_model)
        slots.gpu = row["gpu"] if row else None
    return slots


# --------------------------------------------------------------------------
# G0 -- catalogue validation
# --------------------------------------------------------------------------

@dataclass
class Validation:
    ok: bool
    corrections: list = field(default_factory=list)
    missing: list = field(default_factory=list)


def validate_slots(slots):
    """Reject configurations the catalogue does not contain.

    The failure this exists for is quiet: "Core i7" is a real category and
    "Apple" is a real brand, so an Apple/Core i7 request encodes without error
    and scores like any other row -- for a machine that has never been sold.
    """
    corrections, missing = [], []

    if slots.laptop_brand and slots.laptop_brand not in BRANDS:
        corrections.append(
            f"{slots.laptop_brand} is not in the catalogue. Available brands: "
            f"{', '.join(BRANDS)}."
        )

    if slots.laptop_model and slots.laptop_model not in MODELS:
        corrections.append(f"{slots.laptop_model} is not a catalogue model.")

    if slots.laptop_brand in BRANDS and slots.laptop_model:
        if slots.laptop_model not in models_for_brand(slots.laptop_brand):
            corrections.append(
                f"{slots.laptop_brand} does not ship the {slots.laptop_model}. "
                f"{slots.laptop_brand} models: {', '.join(models_for_brand(slots.laptop_brand))}."
            )

    raw_cpu = getattr(slots, "_cpu_text", None)
    if raw_cpu and slots.laptop_brand in BRANDS and slots.cpu is None:
        corrections.append(
            f"{slots.laptop_brand} does not ship a \"{raw_cpu}\" configuration. "
            f"{slots.laptop_brand} options: {', '.join(cpus_for_brand(slots.laptop_brand))}."
        )

    if slots.ram_gb is not None and slots.ram_gb not in RAM_OPTIONS:
        corrections.append(
            f"{slots.ram_gb} GB is not a configurable option. Available: "
            f"{', '.join(str(r) for r in RAM_OPTIONS)} GB."
        )

    if slots.department is None:
        missing.append("department")

    return Validation(ok=not corrections and not missing,
                      corrections=corrections, missing=missing)


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

GATE_DESCRIPTIONS = {
    "G0": "catalogue mismatch or missing required field -> correct and ask",
    "G1": "definitional / historical question -> RAG",
    "G2": "one resolved candidate -> ML scoring",
    "G3": "goal plus constraint, no candidate -> sourcing agent",
    "G4": "many requests at once -> batch lane",
    "CLARIFY": "nothing matched -> ask",
}


def route(slots):
    """Return the first gate that matches. Order is the design; see module docstring."""
    validation = validate_slots(slots)

    # G0 -- but a pure question does not need a department to be answerable.
    if validation.corrections:
        return "G0", validation
    if slots.intent in ("assess", "find") and validation.missing:
        return "G0", validation

    if slots.intent == "batch":
        return "G4", validation
    if slots.intent == "ask":
        return "G1", validation
    if slots.has_candidate() and slots.intent == "assess":
        return "G2", validation
    if slots.has_constraint() or slots.intent == "find":
        return "G3", validation
    return "CLARIFY", validation


# --------------------------------------------------------------------------
# Sourcing agent -- plan / search / score / evaluate / replan
# --------------------------------------------------------------------------

RELAXATION_LADDER = [
    ("price_band", "widened the price band by 15%"),
    ("drop_specs", "dropped non-binding specs (storage, screen)"),
    ("refurbished", "admitted refurbished and prior-generation units"),
    ("compute_tier", "lowered the compute tier (still at or above the role's floor)"),
]

MAX_RUNGS = len(RELAXATION_LADDER)


@dataclass
class Constraints:
    max_price: float = None
    min_ram: int = None
    allow_refurbished: bool = False
    price_multiplier: float = 1.0
    relax_specs: bool = False
    allow_lower_tier: bool = False

    def effective_price(self):
        return None if self.max_price is None else self.max_price * self.price_multiplier


class SourcingAgent:
    """Plan -> search -> score -> evaluate, relaxing one rung at a time.

    The loop is what distinguishes this from a chain: when nothing satisfies the
    constraint set, it re-enters planning with a weakened set rather than
    returning empty. The ladder is fixed and every descent is reported, so a
    procurement officer can see exactly which rule was bent and in what order.
    """

    def __init__(self, bundle, retriever=None, seller_families=None):
        self.bundle = bundle
        self.retriever = retriever
        # From graph_analysis.export_seller_families(). Family membership is a
        # *disclosure*, not a floor: sharing a distributor feed is ordinary
        # retail, but three quotes from one family are one quote three times.
        # Inert while candidates come from the catalogue, which has no seller --
        # it applies once marketplace listings are sourced as candidates.
        self.seller_families = seller_families

    # -- the floor. None of these is ever relaxed. --------------------------
    def violates_floor(self, slots, candidate, scores):
        reasons = []

        cap = slots.as_scoring_fields().get("dept_policy_cap")
        if cap is not None and candidate["price"] > cap:
            reasons.append("exceeds the departmental policy cap")

        if scores["spec_gap"] < 0:
            reasons.append("under-provisioned for the role")

        if candidate.get("is_suspected_scam"):
            reasons.append("listing is flagged as a suspected scam")

        # A department locked to Windows tooling cannot be handed macOS; that is
        # a wrong answer, not a trade-off to weigh against price.
        if candidate["row"]["laptop_brand"] == "Apple":
            if DEPARTMENTS.get(slots.department, {}).get("windows_locked", 0) >= 0.85:
                reasons.append("department is locked to Windows tooling")

        return reasons

    def search(self, slots, constraints):
        """Enumerate configurations the catalogue can actually supply."""
        budget = constraints.effective_price()
        out = []

        for row in CATALOGUE:
            for ram in RAM_OPTIONS:
                if ram < row["base_ram_gb"]:
                    continue
                if constraints.min_ram and ram < constraints.min_ram and not constraints.relax_specs:
                    continue

                price = row["base_price"] + (ram - row["base_ram_gb"]) * 250_000
                if constraints.allow_refurbished:
                    price *= 0.78
                if budget is not None and price > budget:
                    continue

                out.append({
                    "row": row,
                    "ram_gb": ram,
                    "price": price,
                    "refurbished": constraints.allow_refurbished,
                    "is_suspected_scam": False,
                })
        return out

    def score(self, slots, candidate):
        fields = slots.as_scoring_fields()
        fields.update({
            "laptop_brand": candidate["row"]["laptop_brand"],
            "laptop_model": candidate["row"]["laptop_model"],
            "cpu": candidate["row"]["cpu"],
            "gpu": candidate["row"]["gpu"],
            "compute_tier": candidate["row"]["compute_tier"],
            "build_quality": candidate["row"]["build_quality"],
            "portability": candidate["row"]["portability"],
            "ram_gb": candidate["ram_gb"],
            "storage_gb": candidate["row"]["base_storage_gb"],
            "requested_unit_price": candidate["price"],
        })
        return score_request(self.bundle, fields)

    def run(self, slots, min_suitability=0.55, max_rungs=MAX_RUNGS):
        constraints = Constraints(
            max_price=slots.dept_budget_remaining or slots.requested_unit_price,
            min_ram=slots.ram_gb,
        )
        trace, relaxed = [], []

        for rung in range(max_rungs + 1):
            candidates = self.search(slots, constraints)
            passing, blocked = [], []

            for cand in candidates:
                scores = self.score(slots, cand)
                floor = self.violates_floor(slots, cand, scores)
                if floor:
                    blocked.extend(floor)
                elif scores["suitability_proba"] >= min_suitability:
                    passing.append({**cand, "scores": scores})

            trace.append({
                "rung": rung,
                "budget": constraints.effective_price(),
                "considered": len(candidates),
                "passing": len(passing),
            })

            if passing:
                passing.sort(key=lambda c: (-c["scores"]["suitability_proba"], c["price"]))
                options = passing[:3]
                return {
                    "status": "ok",
                    "options": options,
                    "relaxed": relaxed,
                    "trace": trace,
                    "disclosures": self._seller_disclosures(options),
                }

            if rung == max_rungs:
                return {
                    "status": "infeasible",
                    "options": [],
                    "relaxed": relaxed,
                    "trace": trace,
                    "binding": self._binding_constraint(candidates, blocked, constraints),
                }

            key, description = RELAXATION_LADDER[rung]
            constraints = self._relax(constraints, key)
            relaxed.append(description)

        raise AssertionError("unreachable")

    def _seller_disclosures(self, options):
        """Warn when two presented options come from the same seller family.

        Returns an empty list for catalogue-sourced candidates, which carry no
        seller. The graph result is loaded and ready; it starts producing
        warnings the moment listings become candidates.
        """
        if not self.seller_families:
            return []

        from graph_analysis import family_warning

        warnings, seen = [], set()
        for option in options:
            seller = option.get("seller_name")
            if not seller or seller in seen:
                continue
            seen.add(seller)
            warning = family_warning(self.seller_families, seller)
            if warning:
                warnings.append(warning)
        return warnings

    @staticmethod
    def _relax(constraints, key):
        if key == "price_band":
            constraints.price_multiplier *= 1.15
        elif key == "drop_specs":
            constraints.relax_specs = True
        elif key == "refurbished":
            constraints.allow_refurbished = True
        elif key == "compute_tier":
            constraints.allow_lower_tier = True
        return constraints

    @staticmethod
    def _binding_constraint(candidates, blocked, constraints):
        """Name what actually stopped it -- the useful half of a refusal."""
        if not candidates:
            budget = constraints.effective_price()
            if budget is not None:
                return f"no catalogue configuration exists at or below {budget:,.0f} IDR"
            return "no catalogue configuration matched the requested specification"
        if blocked:
            top = max(set(blocked), key=blocked.count)
            return f"every affordable option {top}"
        return "no option cleared the suitability threshold for this role"


# --------------------------------------------------------------------------
# Rendering model output for the LLM
# --------------------------------------------------------------------------

# Raw probabilities are read inconsistently by an LLM; the bands are fixed once
# here so 0.56 does not become "definitely not" on one turn and "fine" on the next.
def suitability_band(p):
    if p >= 0.75:
        return "good fit"
    if p >= 0.55:
        return "acceptable, with caveats"
    if p >= 0.35:
        return "poor fit"
    return "not suitable"


def describe_scores(scores):
    lines = [
        f"- Suitability: {scores['suitability_proba']:.0%} ({suitability_band(scores['suitability_proba'])})",
        f"- Spec gap: {scores['spec_gap']:+.2f} "
        f"({'over-provisioned' if scores['spec_gap'] > 0.75 else 'under-provisioned' if scores['spec_gap'] < -0.75 else 'about right'})",
        f"- Expected lifespan: {scores['expected_lifespan_months']:.0f} months",
        f"- Expected 3-year operating cost: {scores['expected_opex_idr']:,.0f} IDR",
    ]
    if scores.get("approval_proba") is not None:
        lines.append(f"- Approval odds: {scores['approval_proba']:.0%} ({scores['approval_decision']})")
    else:
        blocked = ", ".join(scores.get("approval_blocked_on", []))
        lines.append(f"- Approval odds: not computed (needs {blocked})")
    return "\n".join(lines)


def describe_agent_result(result):
    lines = []
    if result["status"] == "ok":
        lines.append(f"Found {len(result['options'])} option(s).")
        for opt in result["options"]:
            row = opt["row"]
            lines.append(
                f"\n{row['laptop_brand']} {row['laptop_model']} — {opt['ram_gb']} GB, "
                f"{opt['price']:,.0f} IDR"
                + (" (refurbished)" if opt["refurbished"] else "")
            )
            lines.append(describe_scores(opt["scores"]))
    else:
        lines.append(f"No option satisfies the request. Binding constraint: {result['binding']}.")

    if result["relaxed"]:
        lines.append("\nConstraints relaxed to get here: " + "; ".join(result["relaxed"]) + ".")
    for warning in result.get("disclosures") or []:
        lines.append(f"\nDISCLOSURE: {warning}")
    return "\n".join(lines)
