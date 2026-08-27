"""
Streamlit chatbot UI: retrieves context from Qdrant (collection populated by
embedding.py) and generates answers with Qwen2.5-14B-Instruct via the
Hugging Face Inference API.

Run with: streamlit run app.py
On Kaggle/Colab, start it from a notebook cell and open the ngrok tunnel there
(see the launcher cell) -- never from inside this file, which Streamlit re-runs
on every interaction.
Requires HF_TOKEN in .env with access to Qwen/Qwen2.5-14B-Instruct
(serverless Inference API, or point HF_MODEL at a dedicated Inference
Endpoint URL if the model isn't available serverless).
"""

import os

import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

load_dotenv()

QDRANT_URL = 
QDRANT_API_KEY = 
QDRANT_COLLECTION =
HF_TOKEN = 
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-14B-Instruct")

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"  # must match embedding.py

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about BNI training "
    "documents. Answer using only the provided context. If the answer "
    "isn't in the context, say you don't know instead of guessing."
)

st.set_page_config(page_title="BNI Training Assistant", page_icon="💬")


@st.cache_resource(show_spinner="Connecting to Qdrant...")
def get_index():
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    client = qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION)
    return VectorStoreIndex.from_vector_store(vector_store)


@st.cache_resource
def get_hf_client():
    return InferenceClient(model=HF_MODEL, token=HF_TOKEN)


def build_user_turn(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no matching context found)"
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above."


index = get_index()
hf_client = get_hf_client()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved chunks", 1, 10, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("💬 BNI Training Assistant")
st.caption(f"RAG over Qdrant collection `{QDRANT_COLLECTION}` · {HF_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something about the training documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)
    context_chunks = [n.node.get_content() for n in nodes]

    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    chat_messages.extend(
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    )
    chat_messages.append({"role": "user", "content": build_user_turn(question, context_chunks)})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        try:
            stream = hf_client.chat_completion(
                messages=chat_messages,
                max_tokens=1024,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                delta = getattr(choices[0].delta, "content", None) or ""
                answer += delta
                if answer:
                    placeholder.markdown(answer + "▌")

            if not answer:
                raise RuntimeError(
                    "The model returned no text. Check that HF_MODEL is valid "
                    "and available to your Hugging Face account."
                )
            placeholder.markdown(answer)
        except Exception as e:
            answer = f"Error calling {HF_MODEL}: {e}"
            placeholder.markdown(answer)

        if nodes:
            with st.expander("Sources"):
                for n in nodes:
                    meta = n.node.metadata
                    st.markdown(
                        f"**{meta.get('source', '?')} — page {meta.get('page', '?')}** "
                        f"(score: {n.score:.3f})"
                    )
                    st.text(n.node.get_content()[:500])

    st.session_state.messages.append({"role": "assistant", "content": answer})