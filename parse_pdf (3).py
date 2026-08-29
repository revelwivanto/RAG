"""
Parsing PDF -> tiga kategori (TEXT/IMAGE/TABLE), lalu dikonversi ke skema
chunk parent-child siap RAG:

- TEXT  : parent = paragraf asli utuh, child = potongan (chunk) untuk embedding.
- IMAGE : parent = paragraf teks terdekat sebelumnya, child = caption (dari VLM).
- TABLE : parent = tabel dalam format Markdown, child = ringkasan (dari LLM).

Caption gambar & summary tabel dihasilkan oleh satu model lokal:
Qwen/Qwen3-VL-7B-Instruct (via transformers, butuh GPU ~9GB VRAM Q5 / lebih
untuk bf16 penuh). Kalau gagal load (lib/model tak tersedia), field
caption/summary otomatis None.
"""

import fitz
import camelot
import json
import os
import re

_MODEL_ID = "Qwen/Qwen3-VL-7B-Instruct"
_model = None
_processor = None


def _load_model():
    """Lazy-load Qwen3-VL-7B sekali saja (dipakai bareng untuk caption gambar & summary tabel)."""
    global _model, _processor
    if _model is not None:
        return _model, _processor
    try:
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            _MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        _processor = AutoProcessor.from_pretrained(_MODEL_ID)
    except Exception as e:
        print(f"[warn] gagal load {_MODEL_ID}: {e}")
        _model, _processor = None, None
    return _model, _processor


def _generate(messages, max_new_tokens=200):
    model, processor = _load_model()
    if model is None:
        return None
    try:
        from qwen_vl_utils import process_vision_info

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        result = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return result[0].strip()
    except Exception as e:
        print(f"[warn] generate gagal: {e}")
        return None


def bbox_overlap(bbox_a, bbox_b, threshold=0.5):
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter_area = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    return area_a > 0 and (inter_area / area_a) >= threshold


def bbox_to_dict(bbox):
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    return {"x0": round(x0, 1), "y0": round(y0, 1), "x1": round(x1, 1), "y1": round(y1, 1)}


def extract_tables_per_page(pdf_path, page_number):
    results = []
    try:
        tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor="lattice", line_scale=40)
    except Exception:
        tables = []

    if len(tables) == 0 or all(t.parsing_report.get("accuracy", 0) < 50 for t in tables):
        try:
            stream_tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor="stream",
                                              edge_tol=200, row_tol=10)
            if len(stream_tables) > 0:
                tables = stream_tables
        except Exception:
            pass

    with fitz.open(pdf_path) as doc:
        page_height = doc[page_number - 1].rect.height

    for t in tables:
        x0, y0, x1, y1 = t._bbox
        bbox_pymupdf = (x0, page_height - y1, x1, page_height - y0)
        results.append({
            "bbox": bbox_pymupdf,
            "n_rows": t.shape[0],
            "n_cols": t.shape[1],
            "data": t.data,
            "accuracy": round(t.parsing_report.get("accuracy", 0), 1),
        })
    return results


def extract_images_per_page(doc, page, page_number, images_dir):
    results = []
    seen_xref = set()
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        if xref in seen_xref:
            continue
        seen_xref.add(xref)
        try:
            base_image = doc.extract_image(xref)
        except Exception:
            continue

        ext = base_image["ext"]
        fname = f"page{page_number}_img{img_index+1}.{ext}"
        fpath = os.path.join(images_dir, fname)
        with open(fpath, "wb") as f:
            f.write(base_image["image"])

        try:
            rects = page.get_image_rects(xref)
            bbox = tuple(rects[0]) if rects else None
        except Exception:
            bbox = None

        results.append({
            "bbox": bbox,
            "width": base_image.get("width"),
            "height": base_image.get("height"),
            "path": fpath,
        })
    return results


def bbox_union(boxes):
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def find_chunk_bbox(full_text, lines_info, chunk_str, search_start=0):
    """
    Cari posisi chunk_str di full_text (mulai dari search_start biar chunk
    berikutnya tidak nemu posisi chunk sebelumnya kalau ada teks berulang),
    lalu union bbox baris-baris yang overlap rentang karakter itu.
    Kalau gagal ketemu (mismatch whitespace dll), return None -> fallback ke
    bbox seluruh block di pemanggil.
    """
    idx = full_text.find(chunk_str, search_start)
    if idx == -1:
        return None, search_start
    end = idx + len(chunk_str)
    boxes = [l["bbox"] for l in lines_info if l["end"] >= idx and l["start"] <= end]
    if not boxes:
        return None, search_start
    return bbox_union(boxes), end


def chunk_text(text, max_chars=300):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_chars:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks or [text]


def table_to_markdown(data):
    if not data:
        return ""
    header, rows = data[0], data[1:]
    md = "| " + " | ".join(str(c).strip() for c in header) + " |\n"
    md += "|" + "|".join(["---"] * len(header)) + "|\n"
    for r in rows:
        md += "| " + " | ".join(str(c).strip() for c in r) + " |\n"
    return md


def generate_image_caption(image_path):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": "Describe this image in one concise sentence."},
        ],
    }]
    return _generate(messages, max_new_tokens=100)


def generate_table_summary(md_table):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text":
                f"Summarize this table in 2-3 sentences, highlighting key values/trends:\n\n{md_table}"},
        ],
    }]
    return _generate(messages, max_new_tokens=200)


def parse_pdf(pdf_path, max_pages=None, output_dir="/home/claude/output"):
    doc = fitz.open(pdf_path)
    pdf_metadata = dict(doc.metadata or {})  # title, author, subject, dll (kalau ada di PDF)
    n_pages = len(doc) if max_pages is None else min(max_pages, len(doc))
    output = []
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    for page_index in range(n_pages):
        page = doc[page_index]
        page_number = page_index + 1

        table_blocks = extract_tables_per_page(pdf_path, page_number)
        table_bboxes = [t["bbox"] for t in table_blocks]

        raw_blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        image_blocks = extract_images_per_page(doc, page, page_number, images_dir)

        for block in raw_blocks:
            bbox = block["bbox"]
            if block["type"] == 1:
                continue
            if block["type"] == 0:
                if any(bbox_overlap(bbox, tb, threshold=0.5) for tb in table_bboxes):
                    continue  # sudah tercakup di table_blocks, jangan dobel

                # Simpan bbox per BARIS (bukan cuma bbox seluruh block) supaya
                # nanti highlight per-chunk bisa presisi ke baris yang relevan,
                # bukan seluruh block (yang bisa mencakup beberapa baris/paragraf
                # yang PyMuPDF anggap satu "block" visual).
                lines_info = []
                text_parts = []
                offset = 0
                for line in block.get("lines", []):
                    line_text = "".join(span["text"] for span in line.get("spans", [])).strip()
                    if not line_text:
                        continue
                    start = offset
                    text_parts.append(line_text)
                    offset += len(line_text) + 1  # +1 utk pemisah spasi antar baris
                    lines_info.append({"text": line_text, "bbox": line["bbox"], "start": start, "end": offset - 1})

                text = " ".join(text_parts).strip()
                if not text:
                    continue

                # Teks yang overlap gambar BUKAN duplikat (beda dari kasus tabel) —
                # biasanya ini caption/keterangan gambar, jadi tetap disimpan sebagai
                # text_block biasa, tapi ditandai supaya bisa dikenali sebagai caption.
                overlapping_img = next(
                    (ib for ib in image_blocks if ib["bbox"] and bbox_overlap(bbox, ib["bbox"], threshold=0.3)),
                    None,
                )
                text_blocks.append({
                    "bbox": bbox,
                    "text": text,
                    "lines": lines_info,
                    "overlaps_image_path": overlapping_img["path"] if overlapping_img else None,
                })

        output.append({
            "page": page_number,
            "text_blocks": text_blocks,
            "image_blocks": image_blocks,
            "table_blocks": table_blocks,
            # metrik mentah untuk evaluasi kualitas parsing (lihat evaluate_parsing)
            "page_raw_text_len": len(page.get_text("text").strip()),
            "raw_image_type1_count": sum(1 for b in raw_blocks if b["type"] == 1),
            "page_width": round(page.rect.width, 1),
            "page_height": round(page.rect.height, 1),
            "doc_metadata": pdf_metadata,
        })
        print(f"Halaman {page_number}: {len(text_blocks)} teks, {len(image_blocks)} gambar, {len(table_blocks)} tabel")

    doc.close()
    return output


def to_chunks(pages, doc_id, source_pdf, generate_captions=True, generate_summaries=True):
    """
    Ubah output parse_pdf() jadi flat list record parent-child sesuai skema RAG.
    Urutan elemen per halaman diurutkan berdasarkan posisi vertikal (y0) supaya
    "paragraf sebelumnya" untuk gambar bisa ditentukan dengan benar.

    source_pdf: path lengkap ke file PDF sumber -> disimpan sebagai
    source_pdf_path; basename-nya disimpan sebagai source_pdf.
    doc_title diambil dari metadata PDF (title), fallback ke nama file.
    """
    records = []
    source_pdf_path = source_pdf
    source_pdf_name = os.path.basename(source_pdf)

    raw_title = (pages[0].get("doc_metadata", {}) or {}).get("title") if pages else None
    doc_title = raw_title.strip() if raw_title and raw_title.strip() else os.path.splitext(source_pdf_name)[0]

    for pg in pages:
        page_number = pg["page"]

        elements = (
            [("text", tb) for tb in pg["text_blocks"]]
            + [("image", ib) for ib in pg["image_blocks"]]
            + [("table", tab) for tab in pg["table_blocks"]]
        )
        elements.sort(key=lambda e: e[1]["bbox"][1] if e[1]["bbox"] else 0)

        last_text_id = None
        text_counter = img_counter = tab_counter = 0
        caption_by_image_path = {}

        # Pre-pass: tentukan id tiap paragraf teks lebih dulu, supaya caption yang
        # posisinya SETELAH gambar (mis. di bawah gambar) tetap bisa dipetakan
        # sebelum record gambar-nya di-emit di pass utama.
        _tc = 0
        for kind, el in elements:
            if kind == "text":
                _tc += 1
                if el.get("overlaps_image_path"):
                    caption_by_image_path[el["overlaps_image_path"]] = f"p{page_number}_para{_tc}"

        for kind, el in elements:
            if kind == "text":
                text_counter += 1
                parent_id = f"p{page_number}_para{text_counter}"
                records.append({
                    "doc_id": doc_id,
                    "id": parent_id,
                    "type": "text",
                    "page_start": page_number,
                    "page_end": page_number,
                    "source_pdf": source_pdf_name,
                    "source_pdf_path": source_pdf_path,
                    "doc_title": doc_title,
                    "page_width": pg.get("page_width"),
                    "page_height": pg.get("page_height"),
                    "bounding_box": bbox_to_dict(el["bbox"]),
                    "teks_asli": el["text"],
                    "is_caption": bool(el.get("overlaps_image_path")),
                })
                for j, chunk in enumerate(chunk_text(el["text"])):
                    chunk_bbox, _search_pos = find_chunk_bbox(
                        el["text"], el.get("lines", []), chunk, _search_pos if j else 0
                    )
                    records.append({
                        "doc_id": doc_id,
                        "id": f"{parent_id}_chunk{j+1}",
                        "type": "text_chunk",
                        "parent_id": parent_id,
                        "page_start": page_number,
                        "page_end": page_number,
                        "source_pdf": source_pdf_name,
                        "source_pdf_path": source_pdf_path,
                        "doc_title": doc_title,
                        "page_width": pg.get("page_width"),
                        "page_height": pg.get("page_height"),
                        # bbox presisi ke baris yang relevan; fallback ke bbox
                        # seluruh paragraf kalau gagal dicocokkan (mis. mismatch
                        # whitespace hasil chunk_text vs teks asli baris).
                        "bounding_box": bbox_to_dict(chunk_bbox) if chunk_bbox else bbox_to_dict(el["bbox"]),
                        "id_chunk": f"chunk_{j+1}",
                        "teks": chunk,
                    })
                last_text_id = parent_id

            elif kind == "image":
                img_counter += 1
                image_id = f"p{page_number}_img{img_counter}"
                caption = generate_image_caption(el["path"]) if generate_captions else None
                # Prioritas parent: caption yang overlap langsung dengan gambar ini,
                # baru fallback ke paragraf teks terdekat sebelumnya.
                parent_id = caption_by_image_path.get(el["path"], last_text_id)
                records.append({
                    "doc_id": doc_id,
                    "id": image_id,
                    "type": "image",
                    "page_start": page_number,
                    "page_end": page_number,
                    "source_pdf": source_pdf_name,
                    "source_pdf_path": source_pdf_path,
                    "doc_title": doc_title,
                    "page_width": pg.get("page_width"),
                    "page_height": pg.get("page_height"),
                    "bounding_box": bbox_to_dict(el["bbox"]),
                    "file": el["path"],
                    "parent_id": parent_id,
                })
                records.append({
                    "doc_id": doc_id,
                    "id": f"{image_id}_caption",
                    "type": "image_caption",
                    "parent_id": image_id,
                    "page_start": page_number,
                    "page_end": page_number,
                    "teks": caption,
                })

            elif kind == "table":
                tab_counter += 1
                table_id = f"p{page_number}_table{tab_counter}"
                md = table_to_markdown(el["data"])
                summary = generate_table_summary(md) if (generate_summaries and md) else None
                records.append({
                    "doc_id": doc_id,
                    "id": table_id,
                    "type": "table",
                    "page_start": page_number,
                    "page_end": page_number,
                    "source_pdf": source_pdf_name,
                    "source_pdf_path": source_pdf_path,
                    "doc_title": doc_title,
                    "page_width": pg.get("page_width"),
                    "page_height": pg.get("page_height"),
                    "bounding_box": bbox_to_dict(el["bbox"]),
                    "n_rows": el["n_rows"],
                    "n_cols": el["n_cols"],
                    "markdown": md,
                })
                records.append({
                    "doc_id": doc_id,
                    "id": f"{table_id}_summary",
                    "type": "table_summary",
                    "parent_id": table_id,
                    "page_start": page_number,
                    "page_end": page_number,
                    "teks": summary,
                })

    return records


def evaluate_page(pg, low_table_acc_threshold=70, min_coverage=0.6, scanned_threshold=20):
    """
    Evaluasi kualitas parsing satu halaman. Sinyal yang dicek:
    - text_coverage_ratio: total karakter di text_blocks (setelah difilter tabel/caption)
      dibagi total karakter mentah dari page.get_text("text"). Rendah -> ada teks yang
      "hilang" (mis. salah kefilter sebagai bagian tabel, atau font tak terbaca).
    - likely_scanned: nyaris tidak ada layer teks sama sekali -> kemungkinan hasil scan,
      butuh OCR dulu sebelum PyMuPDF bisa baca apa pun.
    - low_confidence_tables: tabel dengan skor accuracy Camelot di bawah ambang.
    - image_count_mismatch: selisih jumlah gambar dari block type=1 (raw) vs yang
      berhasil diekstrak lewat xref -> indikasi ada gambar yang tidak tertangkap/dobel.
    """
    text_len_extracted = sum(len(tb["text"]) for tb in pg["text_blocks"])
    text_len_raw = pg.get("page_raw_text_len", 0)
    coverage = round(text_len_extracted / text_len_raw, 2) if text_len_raw else None
    likely_scanned = text_len_raw < scanned_threshold

    low_conf_tables = sum(1 for t in pg["table_blocks"] if t["accuracy"] < low_table_acc_threshold)
    image_mismatch = pg.get("raw_image_type1_count", 0) - len(pg["image_blocks"])

    if likely_scanned:
        status = "likely_scanned_needs_ocr"
    elif coverage is not None and coverage < min_coverage:
        status = "needs_review_low_text_coverage"
    elif low_conf_tables > 0:
        status = "needs_review_low_table_accuracy"
    elif image_mismatch != 0:
        status = "needs_review_image_mismatch"
    else:
        status = "ok"

    return {
        "page": pg["page"],
        "text_coverage_ratio": coverage,
        "n_text_blocks": len(pg["text_blocks"]),
        "n_image_blocks": len(pg["image_blocks"]),
        "n_table_blocks": len(pg["table_blocks"]),
        "low_confidence_tables": low_conf_tables,
        "image_count_mismatch": image_mismatch,
        "likely_scanned": likely_scanned,
        "status": status,
    }


def evaluate_parsing(pages, chunks=None):
    """
    Agregat evaluasi seluruh dokumen. `chunks` (hasil to_chunks) opsional -
    kalau diisi, ikut dihitung berapa caption/summary yang gagal digenerate (None).
    """
    per_page = [evaluate_page(pg) for pg in pages]
    flagged = [p for p in per_page if p["status"] != "ok"]

    covs = [p["text_coverage_ratio"] for p in per_page if p["text_coverage_ratio"] is not None]
    avg_coverage = round(sum(covs) / len(covs), 2) if covs else None

    caption_stats = summary_stats = None
    if chunks is not None:
        captions = [c for c in chunks if c["type"] == "image_caption"]
        summaries = [c for c in chunks if c["type"] == "table_summary"]
        caption_stats = {"total": len(captions), "gagal_generate": sum(1 for c in captions if not c["teks"])}
        summary_stats = {"total": len(summaries), "gagal_generate": sum(1 for c in summaries if not c["teks"])}

    report = {
        "n_pages": len(pages),
        "n_pages_flagged": len(flagged),
        "avg_text_coverage_ratio": avg_coverage,
        "total_tables": sum(p["n_table_blocks"] for p in per_page),
        "total_low_confidence_tables": sum(p["low_confidence_tables"] for p in per_page),
        "total_images": sum(p["n_image_blocks"] for p in per_page),
        "caption_stats": caption_stats,
        "table_summary_stats": summary_stats,
        "flagged_pages": flagged,
        "per_page": per_page,
    }
    return report


if __name__ == "__main__":
    PDF_PATH = r"C:\Users\DELL\OneDrive\Documents\BNI\Dataset RAG\Data_sintesis\BA_Laptop_Yoga_SYNTHETIC_FINAL.pdf"
    DOC_ID = "manual_v3"
    OUTPUT_DIR = r"C:\Users\DELL\OneDrive\Documents\BNI\Dataset RAG\Output"

    pages = parse_pdf(PDF_PATH, output_dir=OUTPUT_DIR)
    chunks = to_chunks(pages, doc_id=DOC_ID, source_pdf=PDF_PATH)
    eval_report = evaluate_parsing(pages, chunks=chunks)

    with open(os.path.join(OUTPUT_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUTPUT_DIR, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)

    print("\n=== RINGKASAN ===")
    print(f"Total record        : {len(chunks)}")
    print(f"Halaman ditandai    : {eval_report['n_pages_flagged']}/{eval_report['n_pages']}")
    print(f"Avg text coverage   : {eval_report['avg_text_coverage_ratio']}")
    print(f"Tabel low-confidence: {eval_report['total_low_confidence_tables']}/{eval_report['total_tables']}")
    if eval_report["flagged_pages"]:
        print("Halaman perlu review:")
        for p in eval_report["flagged_pages"]:
            print(f"  - hal {p['page']}: {p['status']}")
