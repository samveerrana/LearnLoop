from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class SourcePermission:
    allowed: bool
    license_id: str | None
    reason: str


def training_permission(url: str) -> SourcePermission:
    host = (urlparse(url).hostname or "").lower()
    if host == "en.wikipedia.org":
        return SourcePermission(True, "CC-BY-SA-4.0", "Wikipedia text is allowlisted with attribution/share-alike metadata")
    return SourcePermission(False, None, "source license is not allowlisted for automatic parameter training")
