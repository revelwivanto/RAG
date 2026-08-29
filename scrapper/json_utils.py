"""Utilitas parsing JSON yang dipakai bareng oleh beberapa adapter
(Tokopedia, Blibli, dst) -- terutama buat situs yang skema API-nya
sering berubah / gak didokumentasikan resmi.
"""

_PRODUCT_MARKER_KEYS = {"name", "product_name", "productName", "title", "itemName"}
_PRICE_MARKER_KEYS = {"price", "price_int", "priceFmt", "price_str", "priceValue", "salePrice"}


def find_product_list(node, _depth=0):
    """Cari list of dict yang 'mirip' kartu produk di mana pun posisinya di JSON.
    Dipakai sebagai fallback kalau nama field top-level API berubah-ubah."""
    if _depth > 12:
        return None
    if isinstance(node, list):
        if node and isinstance(node[0], dict):
            keys = set(node[0].keys())
            if keys & _PRODUCT_MARKER_KEYS and (keys & _PRICE_MARKER_KEYS or "price" in str(keys).lower()):
                return node
        for item in node:
            found = find_product_list(item, _depth + 1)
            if found:
                return found
    elif isinstance(node, dict):
        for v in node.values():
            found = find_product_list(v, _depth + 1)
            if found:
                return found
    return None


def dig(d, *keys, default=None):
    """Ambil nested value dgn aman; kalau bukan dict di tengah jalan, return default."""
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return cur if cur is not None else default


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_int(v):
    try:
        return int(float(str(v).replace(".", "").replace(",", "")
                    if isinstance(v, str) and v.count(",") + v.count(".") > 1 else v))
    except (TypeError, ValueError):
        return None


def parse_sold_text(text):
    if not text:
        return None
    text = str(text).lower().replace("terjual", "").strip()
    try:
        if "rb" in text:
            return int(float(text.replace("rb", "").strip()) * 1000)
        if "k" in text:
            return int(float(text.replace("k", "").strip()) * 1000)
        return int(text)
    except ValueError:
        return None
