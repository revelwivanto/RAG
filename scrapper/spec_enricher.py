"""Lengkapi field spesifikasi yang kosong dari judul dan halaman produk."""
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import requests
import spec_parser
from schema import LaptopListing

SPEC_FIELDS = ("brand", "model", "cpu", "ram_gb", "storage_gb", "gpu", "screen_size_in")
# Kolom yang dicek untuk drop_empty_spec: brand hampir selalu ketebak dari nama
# toko/judul dan screen_size_in bukan spec penentu, jadi baris baru dibuang
# kalau kelima kolom spec inti ini kosong semua.
CORE_SPEC_FIELDS = ("model", "cpu", "ram_gb", "storage_gb", "gpu")
_LABELS = {
    "brand": r"brand|merek",
    "model": r"model|nomor model|model number|sku",
    "cpu": r"cpu|processor|prosesor",
    "ram_gb": r"ram|memory|memori",
    "storage_gb": r"storage|penyimpanan|ssd|hdd|hard ?disk|nvme",
    "gpu": r"gpu|graphics|grafis|vga|kartu grafis",
    "screen_size_in": r"screen|display|layar|ukuran layar",
}


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    })
    return session


def _page_text(url: str, session: requests.Session, timeout: float = 12) -> str:
    if not url or not url.startswith(("http://", "https://")):
        return ""
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return ""
    content = response.text[:1_500_000]
    # Banyak marketplace menyimpan spesifikasi dalam JSON state aplikasi,
    # bukan hanya di application/ld+json. Ambil semua script agar pasangan
    # label-nilai seperti "ram": "16 GB" ikut terbaca.
    blocks = re.findall(r"<script\b[^>]*>(.*?)</script>", content, flags=re.I | re.S)
    structured = []
    for block in blocks:
        try:
            structured.append(json.dumps(json.loads(block), ensure_ascii=False))
        except json.JSONDecodeError:
            # Script yang bukan JSON kadang memuat state inline. Tetap ambil,
            # tetapi batasi ukurannya agar bundle JavaScript tidak mendominasi.
            structured.append(block[:100_000])
    visible_text = re.sub(r"<[^>]+>", " ", content)
    return html.unescape(" ".join(structured) + " " + visible_text)


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _extract_values(text: str) -> dict:
    return {
        "brand": spec_parser.extract_brand(text),
        "model": spec_parser.extract_model(text),
        "cpu": spec_parser.extract_cpu(text),
        "ram_gb": spec_parser.extract_ram_gb(text),
        "storage_gb": spec_parser.extract_storage_gb(text),
        "gpu": spec_parser.extract_gpu(text),
        "screen_size_in": spec_parser.extract_screen_size(text),
    }


def _labelled_spec_text(page_text: str, field: str) -> str:
    """Ambil konteks pendek setelah label spesifikasi untuk hindari produk terkait."""
    label = _LABELS[field]
    pattern = re.compile(rf"(?i)(?:\"?{label}\"?\s*[:=-]|{label}\s+)")
    return " ".join(
        page_text[max(0, match.start() - 30):match.end() + 280]
        for match in pattern.finditer(page_text)
    )


def _fill_missing(listing: LaptopListing, text: str) -> int:
    values = _extract_values(text)
    values["model"] = spec_parser.extract_model(text, listing.brand)
    filled = 0
    for field, value in values.items():
        if _is_missing(getattr(listing, field)) and value is not None:
            setattr(listing, field, value)
            filled += 1
    return filled


def _fill_from_product_page(listing: LaptopListing, page_text: str) -> int:
    """Isi field kosong dari konteks berlabel, lalu fallback ke metadata halaman."""
    filled = 0
    for field in SPEC_FIELDS:
        if not _is_missing(getattr(listing, field)):
            continue
        labelled_text = _labelled_spec_text(page_text, field)
        value = _extract_values(labelled_text).get(field) if labelled_text else None
        if value is not None:
            setattr(listing, field, value)
            filled += 1

    # Fallback untuk JSON-LD / metadata produk yang tidak memiliki label UI.
    if any(_is_missing(getattr(listing, field)) for field in SPEC_FIELDS):
        filled += _fill_missing(listing, page_text)
    return filled


def enrich_listing(listing: LaptopListing, fetch_product_page: bool = False, session: requests.Session | None = None) -> int:
    """Isi hanya field kosong. Judul selalu diprioritaskan daripada halaman produk."""
    filled = _fill_missing(listing, listing.title)
    if fetch_product_page and any(_is_missing(getattr(listing, field)) for field in SPEC_FIELDS):
        filled += _fill_from_product_page(listing, _page_text(listing.url, session or _new_session()))
    return filled


def _apply_gemini(listings: list[LaptopListing], api_key: str | None, model: str | None) -> int:
    """Isi field yang masih kosong pakai Gemini API (lebih presisi dari regex untuk judul rumit)."""
    import gemini_spec_extractor as gemini

    candidates = [item for item in listings if any(_is_missing(getattr(item, field)) for field in SPEC_FIELDS)]
    if not candidates:
        return 0

    titles = [item.title for item in candidates]
    kwargs = {"api_key": api_key}
    if model:
        kwargs["model"] = model
    results = gemini.extract_specs_batch(titles, **kwargs)

    filled = 0
    for item, values in zip(candidates, results):
        for field, value in values.items():
            if value is not None and _is_missing(getattr(item, field)):
                setattr(item, field, value)
                filled += 1
    return filled


def _has_any_spec(listing: LaptopListing) -> bool:
    return any(not _is_missing(getattr(listing, field)) for field in CORE_SPEC_FIELDS)


def enrich_listings(listings: Iterable[LaptopListing], fetch_product_page: bool = False,
                    web_workers: int = 4, use_gemini: bool = False,
                    gemini_api_key: str | None = None, gemini_model: str | None = None,
                    drop_empty_spec: bool = True) -> list[LaptopListing]:
    items = list(listings)
    count = sum(enrich_listing(item) for item in items)

    if use_gemini:
        count += _apply_gemini(items, gemini_api_key, gemini_model)

    candidates = [item for item in items if any(_is_missing(getattr(item, field)) for field in SPEC_FIELDS)]
    if fetch_product_page and candidates:
        # requests.Session tidak aman dipakai lintas thread; setiap tugas membuat
        # sesi sendiri. Batas worker kecil agar tidak membanjiri marketplace.
        def enrich_from_web(item: LaptopListing) -> int:
            return enrich_listing(item, fetch_product_page=True, session=_new_session())

        with ThreadPoolExecutor(max_workers=max(1, web_workers)) as pool:
            count += sum(pool.map(enrich_from_web, candidates))

    dropped = 0
    if drop_empty_spec:
        before = len(items)
        items = [item for item in items if _has_any_spec(item)]
        dropped = before - len(items)

    source = "title"
    if use_gemini:
        source += " + gemini"
    if fetch_product_page:
        source += " + product page"
    print(f"filled {count} missing specification fields from {source}"
          f" (web candidates={len(candidates) if fetch_product_page else 0}, dropped_empty_spec={dropped})")
    return items
