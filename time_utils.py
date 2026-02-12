# time_utils.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


_BERLIN = ZoneInfo("Europe/Berlin")

# Example:
# "veröffentlicht am 30.01.2026, 10:17 Uhr"
_PUBLISHED_RE = re.compile(
    r"veröffentlicht\s+am\s*(\d{2}\.\d{2}\.\d{4}),\s*(\d{2}:\d{2})\s*Uhr",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class PublishedAt:
    raw: str
    dt: datetime  # tz-aware Europe/Berlin

def parse_published_at(text: str) -> PublishedAt | None:
    """
    Extract and parse 'veröffentlicht am DD.MM.YYYY, HH:MM Uhr' from arbitrary text.
    Returns tz-aware datetime in Europe/Berlin or None if not found.
    """
    if not text:
        return None
    m = _PUBLISHED_RE.search(text)
    if not m:
        return None
    date_str, time_str = m.group(1), m.group(2)
    dt = datetime.strptime(f"{date_str}, {time_str}", "%d.%m.%Y, %H:%M").replace(tzinfo=_BERLIN)
    return PublishedAt(raw=m.group(0), dt=dt)
