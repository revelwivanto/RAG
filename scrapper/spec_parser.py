"""Ekstraksi spesifikasi laptop dari judul atau teks halaman produk."""
import re
from typing import Optional

BRANDS = ["ASUS", "LENOVO", "HP", "ACER", "DELL", "MSI", "AXIOO", "MACBOOK", "APPLE", "HUAWEI", "SAMSUNG"]
GPU_PATTERNS = [r"\b(?:NVIDIA\s+)?RTX\s?(?:A)?\d{3,4}(?:\s*(?:TI|SUPER))?\b", r"\b(?:NVIDIA\s+)?GTX\s?\d{3,4}(?:\s*TI)?\b", r"\b(?:INTEL\s+)?ARC\s+[A-Z]\d{3}[A-Z]?\b", r"\bRADEON\s+(?:RX\s*)?\d{3,4}[A-Z0-9]*\b"]

# Setiap entri: (pola dengan grup tangkap kode/varian, formatter -> nama lengkap vendor+family).
# Urutan penting: pola yang lebih spesifik (mis. Core Ultra) harus dicek sebelum yang umum.
_CPU_SPECS = [
    (r"\b(?:INTEL\s+)?CORE\s+ULTRA\s+([3579])[- ]?(\d{3,5}[A-Z0-9]*)\b",
     lambda m: f"Intel Core Ultra {m.group(1)} {m.group(2)}"),
    (r"\b(?:INTEL\s+)?CORE\s+([3579])[- ]?(\d{3,5}[A-Z0-9]*)\b",
     lambda m: f"Intel Core i{m.group(1)}-{m.group(2)}"),
    (r"\bI([3579])[- ]?(\d{3,5}[A-Z]{0,3})\b",
     lambda m: f"Intel Core i{m.group(1)}-{m.group(2)}"),
    (r"\bRYZEN\s+AI\s+([3579])\s*(\d{3,4}[A-Z0-9]*)\b",
     lambda m: f"AMD Ryzen AI {m.group(1)} {m.group(2)}"),
    (r"\bRYZEN\s+([3579])\s*(\d{3,4}[A-Z0-9]*)\b",
     lambda m: f"AMD Ryzen {m.group(1)} {m.group(2)}"),
    (r"\bATHLON(?:\s+(?!\d{1,4}(?:GB|TB)\b)([A-Z0-9-]{3,}))?\b",
     lambda m: "AMD Athlon" + (f" {m.group(1)}" if m.group(1) else "")),
    (r"\bPENTIUM(?:\s+(?!\d{1,4}(?:GB|TB)\b)([A-Z0-9-]{3,}))?\b",
     lambda m: "Intel Pentium" + (f" {m.group(1)}" if m.group(1) else "")),
    (r"\bCELERON(?:\s+(?!\d{1,4}(?:GB|TB)\b)([A-Z0-9-]{3,}))?\b",
     lambda m: "Intel Celeron" + (f" {m.group(1)}" if m.group(1) else "")),
    (r"\b(?:INTEL\s+)?N(\d{3,4})\b", lambda m: f"Intel N{m.group(1)}"),
    (r"\bSNAPDRAGON\s+((?:(?!RAM\b|MEMORY\b|GB\b|TB\b)[A-Z0-9+-]+\s*){1,4})",
     lambda m: f"Qualcomm Snapdragon {_clean(m.group(1))}"),
    (r"\bM([1-4])(?:\s+(PRO|MAX|ULTRA))?\b",
     lambda m: f"Apple M{m.group(1)}" + (f" {m.group(2).title()}" if m.group(2) else "")),
]


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_brand(text: str) -> Optional[str]:
    for brand in BRANDS:
        if re.search(rf"\b{re.escape(brand)}\b", text.upper()):
            return "Apple" if brand in ("MACBOOK", "APPLE") else brand.title()
    return None


def extract_model(text: str, brand: Optional[str] = None) -> Optional[str]:
    # Kode SKU pabrikan lebih stabil daripada nama seri pemasaran untuk deduplikasi.
    ignored_codes = ("RTX", "GTX", "DDR", "LPDDR", "M365")
    for code in re.findall(r"\b[A-Z]{1,4}\d{3,5}[A-Z0-9-]*\b", text.upper()):
        if not code.startswith(ignored_codes) and not re.fullmatch(r"(?:I|N)\d{3,4}[A-Z]*", code):
            return code
    patterns = [r"\b(?:VIVOBOOK|ZENBOOK)\s+(?:\d{1,2}\s+)?[A-Z]{0,3}\d{3,5}[A-Z0-9-]*\b", r"\b(?:ROG\s+(?:STRIX|ZEPHYRUS|FLOW)|TUF\s+GAMING)\s+[A-Z]{0,3}\d{3,5}[A-Z0-9-]*\b", r"\b(?:IDEAPAD|THINKPAD|LOQ|LEGION|YOGA)\s+[A-Z]{0,3}\d{1,5}[A-Z0-9-]*\b", r"\b(?:PAVILION|VICTUS|ELITEBOOK|PROBOOK|OMEN)\s+[A-Z]{0,3}\d{1,5}[A-Z0-9-]*\b", r"\b(?:ASPIRE|NITRO|SWIFT|PREDATOR)\s+[A-Z]{0,3}\d{1,5}[A-Z0-9-]*\b", r"\b(?:INSPIRON|LATITUDE|XPS|ALIENWARE)\s+[A-Z]{0,3}\d{1,5}[A-Z0-9-]*\b", r"\b(?:MODERN|KATANA|CYBORG|THIN)\s+[A-Z]{0,3}\d{1,5}[A-Z0-9-]*\b", r"\bMACBOOK\s+(?:AIR|PRO)(?:\s+\d{2})?\b"]
    for pattern in patterns:
        match = re.search(pattern, text.upper())
        if match:
            return _clean(match.group(0).title())
    return None


def extract_cpu(text: str) -> Optional[str]:
    upper = text.upper()
    for pattern, formatter in _CPU_SPECS:
        match = re.search(pattern, upper)
        if match:
            return _clean(formatter(match))
    return None


def extract_ram_gb(text: str) -> Optional[int]:
    for pattern in [r"\bRAM\s*(?:UP\s+TO\s*)?(\d{1,3})\s*GB\b", r"\b(\d{1,3})\s*GB\s*(?:RAM|DDR[345]|LPDDR[345X]*)\b", r"\bMEMORY\s*(\d{1,3})\s*GB\b", r"\b(4|8|12|16|24|32|48|64)\s*GB\s+(?:/\s*)?(?:\d{3,4}\s*GB|[1-4]\s*TB)\s*(?:SSD|HDD|NVME)?\b"]:
        match = re.search(pattern, text.upper())
        if match:
            return int(match.group(1))
    return None


# Setiap entri: (pola dengan grup tangkap angka kapasitas, pengali ke GB).
# Pola berlabel SSD/HDD/NVME/EMMC dicek dulu (paling akurat), baru fallback
# tanpa label: "1TB" berdiri sendiri tetap storage (RAM laptop tidak pernah
# dalam TB), dan kombinasi "16GB 512GB"/"512GB 16GB" dipisah lewat ukuran RAM
# yang wajar (4/8/12/16/24/32/48/64GB) supaya angka satunya jelas itu storage.
_STORAGE_PATTERNS = [
    (r"\b(?:SSD|HDD|NVME)\s*(\d{1,2})\s*TB\b", 1024),
    (r"\b(\d{1,2})\s*TB\s*(?:SSD|HDD|NVME)\b", 1024),
    (r"\b(?:SSD|HDD|NVME|EMMC)\s*(\d{2,4})\s*GB\b", 1),
    (r"\b(\d{2,4})\s*GB\s*(?:SSD|HDD|NVME|EMMC)\b", 1),
    (r"\b(\d{1,2})\s*TB\b", 1024),
    (r"\b(?:4|8|12|16|24|32|48|64)\s*GB\s*/?\s*(\d{3,4})\s*GB\b", 1),
    (r"\b(\d{3,4})\s*GB\s*/?\s*(?:4|8|12|16|24|32|48|64)\s*GB\b", 1),
]


def extract_storage_gb(text: str) -> Optional[int]:
    upper = text.upper()
    for pattern, multiplier in _STORAGE_PATTERNS:
        match = re.search(pattern, upper)
        if match:
            return int(match.group(1)) * multiplier
    return None


def extract_gpu(text: str) -> Optional[str]:
    for pattern in GPU_PATTERNS:
        match = re.search(pattern, text.upper())
        if match:
            return _clean(match.group(0))
    return None


def extract_screen_size(text: str) -> Optional[float]:
    match = re.search(r"\b(1[0-9](?:\.\d)?)\s*(?:[\"'\u201c\u201d]|INCH|INCI)\b", text.upper())
    return float(match.group(1)) if match else None
