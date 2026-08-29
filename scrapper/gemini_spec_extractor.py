"""Ekstraksi spesifikasi laptop dari judul memakai Gemini API (model kecil/murah).

Dipakai sebagai fallback presisi tinggi ketika spec_parser.py (regex) tidak
berhasil menemukan suatu field dari judul yang tidak beraturan/berantakan.
Butuh environment variable GEMINI_API_KEY.
"""
import json
import os
import time
from typing import Optional

import requests

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-turbo")
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
BATCH_SIZE = 25
SPEC_FIELDS = ("brand", "model", "cpu", "ram_gb", "storage_gb", "gpu", "screen_size_in")

_SPEC_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "brand": {"type": "string", "nullable": True},
        "model": {"type": "string", "nullable": True},
        "cpu": {"type": "string", "nullable": True},
        "ram_gb": {"type": "integer", "nullable": True},
        "storage_gb": {"type": "integer", "nullable": True},
        "gpu": {"type": "string", "nullable": True},
        "screen_size_in": {"type": "number", "nullable": True},
    },
    "required": ["index", *SPEC_FIELDS],
}
_RESPONSE_SCHEMA = {"type": "array", "items": _SPEC_ITEM_SCHEMA}

_PROMPT = """Kamu adalah parser judul listing laptop e-commerce Indonesia. Untuk setiap judul di bawah, ekstrak spesifikasi laptop secara presisi:
- brand: merek pabrikan (mis. Asus, Lenovo, Hp, Acer, Dell, Msi, Axioo, Apple)
- model: kode model/SKU pabrikan (mis. A416MA, ROG Strix G16), bukan nama toko/promo
- cpu: nama prosesor (mis. Intel Core i5-1240P, Ryzen 5 5500U, Apple M2)
- ram_gb: kapasitas RAM dalam GB (angka saja)
- storage_gb: kapasitas penyimpanan dalam GB (konversi TB ke GB, angka saja)
- gpu: nama GPU diskrit jika disebutkan (mis. RTX 4050), null jika tidak disebutkan (integrated graphics tidak dihitung)
- screen_size_in: ukuran layar dalam inci (angka saja)

Jika suatu field tidak bisa dipastikan dari judul, isi null. Jangan menebak/mengarang nilai.
Kembalikan array JSON, satu objek per judul, urut sesuai "index" yang diberikan.

Judul:
{titles_block}"""


def _build_prompt(titles: list[str], start_index: int) -> str:
    lines = "\n".join(f"{start_index + i}: {title}" for i, title in enumerate(titles))
    return _PROMPT.format(titles_block=lines)


def _call_gemini(prompt: str, api_key: str, model: str, timeout: float = 60, max_retries: int = 3) -> list[dict]:
    url = f"{API_BASE}/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _RESPONSE_SCHEMA,
            "temperature": 0,
        },
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            body = response.json()
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini request gagal setelah {max_retries} percobaan: {last_error}")


def extract_specs_batch(titles: list[str], api_key: Optional[str] = None, model: str = DEFAULT_MODEL,
                         batch_size: int = BATCH_SIZE) -> list[dict]:
    """Ekstrak spec untuk banyak judul sekaligus. Urutan hasil mengikuti urutan titles."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY tidak ditemukan di environment variable.")
    if not titles:
        return []

    results: dict[int, dict] = {}
    for start in range(0, len(titles), batch_size):
        chunk = titles[start:start + batch_size]
        prompt = _build_prompt(chunk, start)
        items = _call_gemini(prompt, api_key, model)
        for item in items:
            idx = item.get("index")
            if idx is not None:
                results[idx] = item

    return [
        {field: results.get(i, {}).get(field) for field in SPEC_FIELDS}
        for i in range(len(titles))
    ]
