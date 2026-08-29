from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional


@dataclass
class LaptopListing:
    source: str
    source_id: str
    url: str
    title: str
    brand: Optional[str]
    model: Optional[str]
    cpu: Optional[str]
    ram_gb: Optional[int]
    storage_gb: Optional[int]
    gpu: Optional[str]
    screen_size_in: Optional[float]
    price_idr: Optional[int]
    original_price_idr: Optional[int]
    condition: str  # "new" | "used"
    seller_name: Optional[str]
    seller_rating: Optional[float]
    seller_num_reviews: Optional[int]
    seller_is_official: bool
    sold_count: Optional[int]
    location: Optional[str]
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_suspected_scam: bool = False
    scam_reasons: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
