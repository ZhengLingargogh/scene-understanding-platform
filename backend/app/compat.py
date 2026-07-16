"""Python 3.8+ compatibility helpers."""

from datetime import datetime, timezone

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # Python < 3.11

__all__ = ["UTC", "datetime", "timezone"]
