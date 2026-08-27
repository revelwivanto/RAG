"""Streamlit UI for the procurement assistant.

One chat box, three lanes behind it. Each turn is extracted into a structured
request, validated against the catalogue, and routed:

    G0  catalogue mismatch / missing field  -> correct and ask, no model call
    G1  question about the documents        -> RAG over Qdrant
    G2  one resolved candidate              -> ML scoring, narrated
    G3  goal plus constraint, no candidate  -> sourcing agent (plan/search/replan)
    G4  many requests at once               -> batch lane
    --  nothing matched                     -> clarify

The routing and agent logic live in procurement_agent.py so this file and the
notebook cannot drift apart -- the two used to declare their own config and
ended up pointing at different Qdrant clusters.

Run with: streamlit run app.py
On Kaggle, start it from a notebook cell and open the tunnel there.
"""

import os

import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from ml_models import load_all, score_request
from procurement_agent import (
    GATE_DESCRIPTIONS, RequestSlots, SourcingAgent,
    describe_agent_result, describe_scores, extract_slots, route,
)

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "capstone")
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
HF_TOKEN = os.environ.get("HF_TOKEN")
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local").lower()
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "http://localhost:8000/v1")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
# bge-m3 embeds one short query per turn, which the CPU handles in well under a
# second. Keeping it off the GPU leaves the whole card to vLLM, which sizes its
# KV cache once at startup and cannot give memory back.
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "cpu")

SYSTEM_PROMPT = (
    "You are a procurement assistant for BNI. You answer about training notes, "
    "laptop procurement documents, and marketplace listings.\n"
    "When the message includes a MODEL OUTPUT block, those numbers come from "
    "trained models and are authoritative -- explain them, never recompute or "
    "contradict them. When it includes a CONTEXT block, answer only from it and "
    "say you don't know if the answer isn't there.\n"
    "Be concise. Reply in the same language as the question."
)

st.set_page_config(page_title="Procurement Assistant", page_icon="🔀", layout="centered")


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model and connecting to Qdrant...")
def get_index():
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME, device=EMBED_DEVICE)
    client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    return VectorStoreIndex.from_vector_store(store)


@st.cache_resource(show_spinner="Loading procurement models...")
def get_bundle():
    return load_all(MODEL_DIR)


@st.cache_resource
def get_llm_client():
    if LLM_BACKEND == "local":
        from openai import OpenAI

        # vLLM ignores the key but the client refuses to construct without one.
        return OpenAI(base_url=LOCAL_LLM_URL, api_key="EMPTY", timeout=600)

    from openai import OpenAI

    return OpenAI(api_key=HF_TOKEN, base_url="https://api-inference.huggingface.co/v1")


def stream_chat(client, messages, temperature):
    stream = client.chat.completions.create(
        model=HF_MODEL, messages=messages, max_tokens=900,
        temperature=temperature, stream=True,
    )
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if choices:
            yield getattr(choices[0].delta, "content", None) or ""


# --------------------------------------------------------------------------
# RAG helpers
# --------------------------------------------------------------------------

def expand_to_parents(nodes):
    """Trade each retrieved sentence for the paragraph it came from.

    The BNI notes are indexed hierarchically: a sentence is embedded as the
    child with its whole paragraph carried in `parent_text`. Matching happens on
    the sentence; the LLM should read the paragraph. Sentences from one
    paragraph collapse so the same text is not sent twice.
    """
    chunks, seen = [], set()
    for scored in nodes:
        node = scored.node
        parent = node.metadata.get("parent_text")
        if parent is None:
            chunks.append(node.get_content())
            continue
        key = node.metadata.get("parent_id", parent)
        if key not in seen:
            seen.add(key)
            chunks.append(parent)
    return chunks


def format_source_label(meta):
    source = meta.get("source", "?")
    kind = meta.get("doc_type")
    if kind == "bni_training":
        return f"{source} — paragraph {meta.get('parent_index', '?')}"
    if kind == "laptop_listing":
        return f"{source} — {meta.get('brand', '?')} listing {meta.get('source_id', '?')}"
    if kind == "purchase_request":
        return f"{source} — request {meta.get('request_id', '?')}"
    if meta.get("page") is not None:
        return f"{source} — page {meta['page']}"
    return source


def retrieve(index, question, top_k):
    nodes = index.as_retriever(similarity_top_k=top_k).retrieve(question)
    return nodes, expand_to_parents(nodes)


# --------------------------------------------------------------------------
# Lanes
# --------------------------------------------------------------------------

def lane_g0(validation):
    """No model call: the catalogue already knows this configuration is wrong."""
    parts = list(validation.corrections)
    if validation.missing:
        parts.append("Still needed: " + ", ".join(validation.missing) + ".")
    return "TASK: relay these catalogue corrections and ask for what is missing.\n\n" + "\n".join(parts)


def lane_g1(index, question, top_k):
    nodes, chunks = retrieve(index, question, top_k)
    context = "\n\n---\n\n".join(chunks) if chunks else "(no matching context found)"
    return f"CONTEXT:\n{context}\n\nQuestion: {question}", nodes


def lane_g2(bundle, slots):
    scores = score_request(bundle, slots.as_scoring_fields())
    block = describe_scores(scores)
    ask = ""
    if scores.get("approval_proba") is None:
        ask = ("\nThe approval model needs "
               + ", ".join(scores.get("approval_blocked_on", []))
               + " -- ask the user for it before promising approval odds.")
    return (f"MODEL OUTPUT for the requested machine:\n{block}\n{ask}\n\n"
            f"Explain this to the user. If the spec gap is positive and large, say the "
            f"configuration is over-specified for the role.\n\nUser said: {slots.free_text}"), scores


def lane_g3(bundle, slots, index, top_k):
    result = SourcingAgent(bundle).run(slots)
    block = describe_agent_result(result)
    nodes, chunks = ([], [])
    if index is not None:
        try:
            nodes, chunks = retrieve(index, slots.free_text, top_k)
        except Exception:
            pass
    context = ("\n\nRELATED PAST PURCHASES / LISTINGS:\n" + "\n\n".join(chunks[:3])) if chunks else ""
    return (f"MODEL OUTPUT from the sourcing agent:\n{block}{context}\n\n"
            f"Present the options (or explain why none exists). State plainly which "
            f"constraints were relaxed.\n\nUser said: {slots.free_text}"), result, nodes


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

index = get_index()
bundle = get_bundle()
llm = get_llm_client()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "slots" not in st.session_state:
    # Slots accumulate across turns: the laptop arrives in one turn and the
    # budget in another. One home for the object, or the router reads a
    # half-built request and mis-gates it.
    st.session_state.slots = RequestSlots()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved chunks", 1, 10, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    show_trace = st.checkbox("Show routing trace", value=True)
    if st.button("Clear chat and request"):
        st.session_state.messages = []
        st.session_state.slots = RequestSlots()
        st.rerun()

    st.divider()
    st.caption("Request under construction")
    known = {k: v for k, v in vars(st.session_state.slots).items()
             if v is not None and not k.startswith("_") and k != "free_text"}
    st.json(known or {"(empty)": None}, expanded=True)

st.title("🔀 Procurement Assistant")
st.caption(f"`{QDRANT_COLLECTION}` · {HF_MODEL} · router + 4 ML models")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Contoh: beli Macbook ram 32gb buat AMGR Design")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        nodes, extra = [], None

        try:
            with st.spinner("Reading the request..."):
                slots = extract_slots(llm, HF_MODEL, question, st.session_state.slots)
                st.session_state.slots = slots
                gate, validation = route(slots)

            if gate == "G0":
                user_turn = lane_g0(validation)
            elif gate == "G1":
                user_turn, nodes = lane_g1(index, question, top_k)
            elif gate == "G2":
                user_turn, extra = lane_g2(bundle, slots)
            elif gate == "G3":
                with st.spinner("Sourcing agent working..."):
                    user_turn, extra, nodes = lane_g3(bundle, slots, index, top_k)
            elif gate == "G4":
                user_turn = ("TASK: tell the user batch review is not wired up yet, "
                             "and offer to assess one request at a time.")
            else:
                user_turn = (f"TASK: ask what the user needs. Known so far: "
                             f"{slots.as_scoring_fields() or 'nothing'}.")

            chat = [{"role": "system", "content": SYSTEM_PROMPT}]
            chat += [{"role": m["role"], "content": m["content"]}
                     for m in st.session_state.messages[-7:-1]]
            chat.append({"role": "user", "content": user_turn})

            answer = ""
            for delta in stream_chat(llm, chat, temperature):
                answer += delta
                if answer:
                    placeholder.markdown(answer + "▌")
            placeholder.markdown(answer or "_(no response)_")

        except Exception as e:
            gate = locals().get("gate", "?")
            answer = f"Error in lane {gate}: {type(e).__name__}: {e}"
            placeholder.markdown(answer)

        if show_trace:
            with st.expander(f"Routing trace — {gate}", expanded=False):
                st.caption(GATE_DESCRIPTIONS.get(gate, ""))
                if extra and isinstance(extra, dict) and "trace" in extra:
                    st.caption("Sourcing agent rungs")
                    st.dataframe(extra["trace"], use_container_width=True)
                    if extra.get("relaxed"):
                        st.caption("Relaxed: " + "; ".join(extra["relaxed"]))
                    if extra.get("binding"):
                        st.warning("Binding constraint: " + extra["binding"])
                elif extra:
                    st.json({k: v for k, v in extra.items() if k != "assumed_fields"})
                    if extra.get("assumed_fields"):
                        st.caption("Assumed (not stated by the user): "
                                   + ", ".join(extra["assumed_fields"]))

        if nodes:
            with st.expander("Sources"):
                for n in nodes:
                    meta = n.node.metadata
                    st.markdown(f"**{format_source_label(meta)}** (score: {n.score:.3f})")
                    parent = meta.get("parent_text")
                    st.text((parent or n.node.get_content())[:900])

    st.session_state.messages.append({"role": "assistant", "content": answer})
