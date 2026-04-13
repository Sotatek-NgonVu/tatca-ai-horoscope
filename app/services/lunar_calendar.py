"""
Solar-to-Lunar calendar conversion for Vietnamese/Chinese calendar.

Uses a lookup-table approach covering years 1900–2100. Each year's lunar
data is encoded as a compact integer bitmask.

Reference: Algorithm adapted from Hồ Ngọc Đức's Vietnamese calendar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ─── Constants ───────────────────────────────────────────────────────────────

_LUNAR_INFO: list[int] = [
    0x04BD8, 0x04AE0, 0x0A570, 0x054D5, 0x0D260,
    0x0D950, 0x16554, 0x056A0, 0x09AD0, 0x055D2,
    0x04AE0, 0x0A5B6, 0x0A4D0, 0x0D250, 0x1D255,
    0x0B540, 0x0D6A0, 0x0ADA2, 0x095B0, 0x14977,
    0x04970, 0x0A4B0, 0x0B4B5, 0x06A50, 0x06D40,
    0x1AB54, 0x02B60, 0x09570, 0x052F2, 0x04970,
    0x06566, 0x0D4A0, 0x0EA50, 0x06E95, 0x05AD0,
    0x02B60, 0x186E3, 0x092E0, 0x1C8D7, 0x0C950,
    0x0D4A0, 0x1D8A6, 0x0B550, 0x056A0, 0x1A5B4,
    0x025D0, 0x092D0, 0x0D2B2, 0x0A950, 0x0B557,
    0x06CA0, 0x0B550, 0x15355, 0x04DA0, 0x0A5D0,
    0x14573, 0x052D0, 0x0A9A8, 0x0E950, 0x06AA0,
    0x0AEA6, 0x0AB50, 0x04B60, 0x0AAE4, 0x0A570,
    0x05260, 0x0F263, 0x0D950, 0x05B57, 0x056A0,
    0x096D0, 0x04DD5, 0x04AD0, 0x0A4D0, 0x0D4D4,
    0x0D250, 0x0D558, 0x0B540, 0x0B6A0, 0x195A6,
    0x095B0, 0x049B0, 0x0A974, 0x0A4B0, 0x0B27A,
    0x06A50, 0x06D40, 0x0AF46, 0x0AB60, 0x09570,
    0x04AF5, 0x04970, 0x064B0, 0x074A3, 0x0EA50,
    0x06B58, 0x05AC0, 0x0AB60, 0x096D5, 0x092E0,
    0x0C960, 0x0D954, 0x0D4A0, 0x0DA50, 0x07552,
    0x056A0, 0x0ABB7, 0x025D0, 0x092D0, 0x0CAB5,
    0x0A950, 0x0B4A0, 0x0BAA4, 0x0AD50, 0x055D9,
    0x04BA0, 0x0A5B0, 0x15176, 0x052B0, 0x0A930,
    0x07954, 0x06AA0, 0x0AD50, 0x05B52, 0x04B60,
    0x0A6E6, 0x0A4E0, 0x0D260, 0x0EA65, 0x0D530,
    0x05AA0, 0x076A3, 0x096D0, 0x04AFB, 0x04AD0,
    0x0A4D0, 0x1D0B6, 0x0D250, 0x0D520, 0x0DD45,
    0x0B5A0, 0x056D0, 0x055B2, 0x049B0, 0x0A577,
    0x0A4B0, 0x0AA50, 0x1B255, 0x06D20, 0x0ADA0,
    0x14B63, 0x09370, 0x049F8, 0x04970, 0x064B0,
    0x168A6, 0x0EA50, 0x06AA0, 0x1A6C4, 0x0AAE0,
    0x092E0, 0x0D2E3, 0x0C960, 0x0D557, 0x0D4A0,
    0x0DA50, 0x05D55, 0x056A0, 0x0A6D0, 0x055D4,
    0x052D0, 0x0A9B8, 0x0A950, 0x0B4A0, 0x0B6A6,
    0x0AD50, 0x055A0, 0x0ABA4, 0x0A5B0, 0x052B0,
    0x0B273, 0x06930, 0x07337, 0x06AA0, 0x0AD50,
    0x14B55, 0x04B60, 0x0A570, 0x054E4, 0x0D160,
    0x0E968, 0x0D520, 0x0DAA0, 0x16AA6, 0x056D0,
    0x04AE0, 0x0A9D4, 0x0A2D0, 0x0D150, 0x0F252,
    0x0D520,
]

_BASE_YEAR = 1900
_BASE_JD = 2415021  # Julian day for Jan 31 1900


@dataclass(frozen=True)
class LunarDate:
    """Represents a Vietnamese/Chinese lunar date."""
    year: int
    month: int
    day: int
    is_leap_month: bool


def _year_days(year_idx: int) -> int:
    """Total days in a lunar year."""
    info = _LUNAR_INFO[year_idx]
    total = 348  # base: 12 months × 29 days
    for i in range(12):
        if info & (0x10000 >> i):
            total += 1
    leap = _leap_month(year_idx)
    if leap:
        total += 30 if info & 0x10000 else 29
    return total


def _leap_month(year_idx: int) -> int:
    """Return leap month number (1-12) or 0 if none."""
    return _LUNAR_INFO[year_idx] & 0xF


def _month_days(year_idx: int, month: int) -> int:
    """Days in a regular (non-leap) month of a lunar year."""
    return 30 if _LUNAR_INFO[year_idx] & (0x10000 >> (month - 1)) else 29


def _leap_month_days(year_idx: int) -> int:
    """Days in the leap month (0 if no leap month)."""
    if not _leap_month(year_idx):
        return 0
    return 30 if _LUNAR_INFO[year_idx] & 0x10000 else 29


def _solar_to_jd(year: int, month: int, day: int) -> int:
    """Convert Gregorian date to Julian Day Number."""
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045


def solar_to_lunar(year: int, month: int, day: int) -> LunarDate:
    """
    Convert a Gregorian (solar) date to Vietnamese lunar date.

    Supports years 1900–2100.
    """
    if year < _BASE_YEAR or year > _BASE_YEAR + len(_LUNAR_INFO) - 1:
        raise ValueError(f"Year {year} out of supported range ({_BASE_YEAR}–{_BASE_YEAR + len(_LUNAR_INFO) - 1})")

    jd = _solar_to_jd(year, month, day)
    # Jan 31, 1900 = Lunar Jan 1, 1900
    offset = jd - _solar_to_jd(1900, 1, 31)

    # Walk through lunar years to find the target year
    lunar_year = _BASE_YEAR
    total = 0
    for i in range(len(_LUNAR_INFO)):
        yd = _year_days(i)
        if total + yd > offset:
            break
        total += yd
        lunar_year += 1
    else:
        raise ValueError("Date out of supported range")

    year_idx = lunar_year - _BASE_YEAR
    remaining = offset - total

    # Walk through months in the target lunar year
    leap = _leap_month(year_idx)
    is_leap = False
    lunar_month = 1

    for m in range(1, 13):
        md = _month_days(year_idx, m)
        if remaining < md:
            lunar_month = m
            break
        remaining -= md

        # Check if this month has a following leap month
        if m == leap:
            ld = _leap_month_days(year_idx)
            if remaining < ld:
                lunar_month = m
                is_leap = True
                break
            remaining -= ld
    else:
        lunar_month = 12

    lunar_day = remaining + 1

    return LunarDate(
        year=lunar_year,
        month=lunar_month,
        day=lunar_day,
        is_leap_month=is_leap,
    )
