"""
Script untuk memparsing PDF dan memisahkan konten menjadi tiga kategori:
1. TEXT  -> paragraf/teks naratif biasa
2. IMAGE -> gambar/foto yang tertanam di halaman
3. TABLE -> data tabular (dideteksi lewat garis/border memakai Camelot)

Cara kerja singkat:
- PyMuPDF (fitz) membaca setiap halaman dan mengembalikan daftar "blocks".
  Setiap block punya field "type": 0 = teks, 1 = gambar. Ini akurat karena
  berdasarkan struktur internal PDF, bukan tebakan visual.
- Camelot mendeteksi tabel lewat pola garis (mode "lattice"). Bounding box
  tabel yang ditemukan dibandingkan dengan blok teks untuk membuang blok
  teks yang sebenarnya "milik" tabel (supaya tidak dobel/duplikat).
- Hasil akhir per halaman disimpan sebagai dictionary terstruktur yang siap
  dipakai untuk tahap chunking & embedding selanjutnya.
"""

import fitz  # PyMuPDF
import camelot
import json
import os

def bbox_overlap(bbox_a, bbox_b, threshold=0.5):
    """Cek apakah dua bounding box overlap signifikan (>= threshold dari area A)."""
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)

    if ix1 <= ix0 or iy1 <= iy0:
        return False  # tidak ada irisan sama sekali

    inter_area = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    if area_a == 0:
        return False
    return (inter_area / area_a) >= threshold


def extract_tables_per_page(pdf_path, page_number):
    """
    Jalankan Camelot di satu halaman (1-indexed sesuai konvensi Camelot).
    Return list of dict: {"bbox": (x0,y0,x1,y1) dalam koordinat PyMuPDF, "data": [[...]]}
    """
    results = []
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor="lattice")
    except Exception:
        tables = []

    page_height = None
    with fitz.open(pdf_path) as doc:
        page_height = doc[page_number - 1].rect.height

    for t in tables:
        # Camelot pakai origin kiri-BAWAH, PyMuPDF pakai origin kiri-ATAS.
        # Perlu konversi sumbu Y supaya bbox bisa dibandingkan dengan blok teks.
        x0, y0, x1, y1 = t._bbox  # (left, bottom, right, top) dalam sistem Camelot
        bbox_pymupdf = (x0, page_height - y1, x1, page_height - y0)
        results.append({
            "bbox": bbox_pymupdf,
            "n_rows": t.shape[0],
            "n_cols": t.shape[1],
            "data": t.data,          # isi tabel mentah, list of list of string
            "accuracy": round(t.parsing_report.get("accuracy", 0), 1),
        })
    return results


def parse_pdf(pdf_path, max_pages=None):
    """
    Parsing utama. Mengembalikan list of dict, satu dict per halaman:
    {
      "page": int,
      "text_blocks": [{"bbox": [...], "text": "..."}],
      "image_blocks": [{"bbox": [...], "width": int, "height": int}],
      "table_blocks": [{"bbox": [...], "n_rows": int, "n_cols": int, "data": [[...]]}]
    }
    """
    doc = fitz.open(pdf_path)
    n_pages = len(doc) if max_pages is None else min(max_pages, len(doc))
    output = []

    for page_index in range(n_pages):
        page = doc[page_index]
        page_number = page_index + 1  # 1-indexed untuk laporan

        # 1. Deteksi tabel dulu di halaman ini (supaya bbox-nya bisa dipakai
        #    untuk memfilter blok teks yang sebenarnya bagian dari tabel).
        table_blocks = extract_tables_per_page(pdf_path, page_number)
        table_bboxes = [t["bbox"] for t in table_blocks]

        # 2. Ambil semua block mentah dari PyMuPDF (teks + gambar sekaligus).
        raw_blocks = page.get_text("dict")["blocks"]

        text_blocks = []
        image_blocks = []

        for block in raw_blocks:
            bbox = block["bbox"]

            if block["type"] == 1:
                # --- GAMBAR ---
                image_blocks.append({
                    "bbox": [round(v, 1) for v in bbox],
                    "width": block.get("width"),
                    "height": block.get("height"),
                })
                continue

            if block["type"] == 0:
                # --- TEKS (sementara), cek dulu apakah overlap dengan tabel ---
                is_part_of_table = any(
                    bbox_overlap(bbox, tb, threshold=0.5) for tb in table_bboxes
                )
                if is_part_of_table:
                    continue  # sudah tercakup di table_blocks, jangan dobel

                text = "".join(
                    span["text"]
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ).strip()

                if text:  # buang blok teks kosong
                    text_blocks.append({
                        "bbox": [round(v, 1) for v in bbox],
                        "text": text,
                    })

        output.append({
            "page": page_number,
            "text_blocks": text_blocks,
            "image_blocks": image_blocks,
            "table_blocks": table_blocks,
        })

        print(f"Halaman {page_number}: "
              f"{len(text_blocks)} blok teks, "
              f"{len(image_blocks)} gambar, "
              f"{len(table_blocks)} tabel")

    doc.close()
    return output


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(DATA_DIR, "parsed_result")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf"))

    if not pdf_files:
        print(f"Tidak ada file PDF ditemukan di {DATA_DIR}")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        print(f"\n=== Memproses: {pdf_file} ===")

        result = parse_pdf(pdf_path)

        out_name = os.path.splitext(pdf_file)[0] + ".json"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        total_text = sum(len(p["text_blocks"]) for p in result)
        total_image = sum(len(p["image_blocks"]) for p in result)
        total_table = sum(len(p["table_blocks"]) for p in result)

        print(f"--- Ringkasan {pdf_file} ---")
        print(f"Total halaman diproses : {len(result)}")
        print(f"Total blok teks        : {total_text}")
        print(f"Total blok gambar      : {total_image}")
        print(f"Total blok tabel       : {total_table}")
        print(f"Hasil disimpan         : {out_path}")
