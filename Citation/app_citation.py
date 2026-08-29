"""
Halaman utama: jawaban bot + citation [1][2].. Klik citation -> buka TAB BARU
ke halaman "Sumber" yang render PDF asli + highlight kalimat yang disitasi.

Struktur wajib (Streamlit multipage):
    app.py
    pages/
        1_Sumber.py

Install: pip install streamlit streamlit-pdf-viewer --break-system-packages
Jalankan: streamlit run app.py
"""

import json
import os
import streamlit as st

st.set_page_config(page_title="Tanya Dokumen", layout="wide")

CHUNKS_PATH = "output/chunks.json"


@st.cache_data
def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def citation_link(n, chunk_id):
    # target="_blank" -> buka tab baru ke halaman "Sumber" (pages/1_Sumber.py)
    return f'<a href="/Sumber?cite={chunk_id}" target="_blank" style="text-decoration:none; color:#2563eb; font-weight:600;">[{n}]</a>'


if not os.path.exists(CHUNKS_PATH):
    st.error(f"{CHUNKS_PATH} tidak ditemukan. Jalankan parse_pdf.py dulu.")
    st.stop()

chunks = load_chunks(CHUNKS_PATH)

st.subheader("Jawaban")

# --- Contoh. Di produksi: teks + retrieved_chunk_ids datang dari LLM + retriever. ---
text_chunks = [c for c in chunks if c["type"] == "text_chunk"]
if not text_chunks:
    st.warning("Tidak ada text_chunk di chunks.json.")
    st.stop()

cid = text_chunks[0]["id"]
answer_html = f"Materi ini membahas perbandingan Hadoop vs Modern Lakehouse {citation_link(1, cid)}."
st.markdown(answer_html, unsafe_allow_html=True)
