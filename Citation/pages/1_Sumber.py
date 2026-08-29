"""
Halaman "Sumber": render halaman PDF jadi gambar (PyMuPDF) + gambar kotak
outline merah (PIL, no-fill) di posisi bbox kalimat yang disitasi.

Ganti dari streamlit-pdf-viewer -> render manual, karena library itu cuma
support fill/solid color, tidak ada opsi outline-only seperti yang diminta.

Install: pip install streamlit pymupdf pillow --break-system-packages
"""

import io
import json
import os
import fitz
from PIL import Image, ImageDraw
import streamlit as st

st.set_page_config(page_title="Dokumen Sumber", layout="wide")

CHUNKS_PATH = "output/chunks.json"
PDF_DIR = "output/source_pdfs"
ZOOM = 2.0                    # skala render (makin besar = makin tajam)
HIGHLIGHT_COLOR = (255, 235, 59, 110)  # kuning, alpha ~110/255 (semi-transparan kayak stabilo)


@st.cache_data
def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_page_with_highlight(pdf_path, page_number, bbox, zoom=ZOOM):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    base = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")

    # overlay terpisah supaya highlight semi-transparan (kayak stabilo asli,
    # teks di bawahnya tetap kebaca), bukan kotak solid nutup teks.
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = bbox["x0"] * zoom, bbox["y0"] * zoom, bbox["x1"] * zoom, bbox["y1"] * zoom
    draw.rectangle([x0, y0, x1, y1], fill=HIGHLIGHT_COLOR)

    result = Image.alpha_composite(base, overlay).convert("RGB")
    doc.close()
    return result


if not os.path.exists(CHUNKS_PATH):
    st.error(f"{CHUNKS_PATH} tidak ditemukan.")
    st.stop()

chunks = load_chunks(CHUNKS_PATH)
chunk_by_id = {c["id"]: c for c in chunks}

cite_id = st.query_params.get("cite")
if not cite_id or cite_id not in chunk_by_id:
    st.warning("Tidak ada citation yang dipilih. Buka halaman ini lewat link [1] di jawaban.")
    st.stop()

chunk = chunk_by_id[cite_id]

# text_chunk sekarang sudah punya bounding_box presisi sendiri (lihat
# find_chunk_bbox di parse_pdf.py) -> pakai langsung, tidak perlu fallback
# ke bbox paragraf induk lagi.
target = chunk

pdf_path = os.path.join(PDF_DIR, target["source_pdf"])

st.subheader(target.get("doc_title", target["source_pdf"]))
st.caption(f"Halaman {target['page_start']}")

if not os.path.exists(pdf_path):
    st.error(f"File PDF tidak ditemukan di: {pdf_path}")
else:
    image = render_page_with_highlight(pdf_path, target["page_start"], target["bounding_box"])
    st.image(image, width=750)
