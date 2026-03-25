#!/usr/bin/env python3
"""Shared time parsing and formatting helpers."""

import datetime as dt
import re


TIME_WITH_OPTIONAL_FRACTION_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})(?:\.(\d{1,6}))?$")


def parse_hhmmss_fractional(time_str):
    """Parse HHMMSS(.fraction) into components."""
    m = TIME_WITH_OPTIONAL_FRACTION_RE.fullmatch(str(time_str).strip())
    if not m:
        raise ValueError("time must be HHMMSS or HHMMSS.s (up to 6 fractional digits)")
    hour = int(m.group(1))
    minute = int(m.group(2))
    second = int(m.group(3))
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError("time components out of range (HH: 00-23, MM/SS: 00-59)")
    frac = m.group(4)
    microsecond = int(frac.ljust(6, "0")) if frac else 0
    return hour, minute, second, microsecond, frac


def parse_date_and_time(date_str, time_str):
    """Parse YYYYMMDD and HHMMSS(.fraction) into a datetime."""
    if not re.fullmatch(r"\d{8}", str(date_str).strip()):
        raise ValueError("date must be YYYYMMDD")
    base_date = dt.datetime.strptime(date_str, "%Y%m%d")
    hour, minute, second, microsecond, _ = parse_hhmmss_fractional(time_str)
    return base_date.replace(hour=hour, minute=minute, second=second, microsecond=microsecond)


def hhmmss_fractional_to_seconds(time_str):
    """Convert HHMMSS(.fraction) to seconds since midnight."""
    hour, minute, second, microsecond, _ = parse_hhmmss_fractional(time_str)
    return hour * 3600.0 + minute * 60.0 + second + microsecond / 1e6


def format_time_label(time_str):
    """Format HHMMSS(.fraction) as HH:MM:SS(.fraction)."""
    hour, minute, second, _, frac = parse_hhmmss_fractional(time_str)
    base = f"{hour:02d}:{minute:02d}:{second:02d}"
    return f"{base}.{frac}" if frac else base


def sanitize_time_for_filename(time_str):
    """Return a filename-safe token preserving fractional seconds."""
    return str(time_str).replace(".", "p")
