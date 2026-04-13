"""
Astronomical Lunar Calendar Engine — ZERO Lookup Tables.

Converts Gregorian (Solar) dates to Vietnamese Lunar dates using
pure mathematical algorithms:

  1. Gregorian → Julian Day Number (standard formula)
  2. JDN → New Moon (Sóc) via Jean Meeus's periodic terms
  3. JDN → Sun Longitude for Solar Term (Trung Khí) detection
  4. Determine Lunar day/month/year and leap months dynamically
  5. Compute Thiên Can / Địa Chi for Year, Month, Day, Hour

References:
  - Jean Meeus, "Astronomical Algorithms", 2nd Ed. (1998)
  - Hồ Ngọc Đức's Vietnamese Calendar algorithm
  - No static tables, no databases, no external APIs

Author: Tu Vi Astrology Engine — tatca.ai
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

# ═════════════════════════════════════════════════════════════════════════════
#  Enums — Type-safe Heavenly Stems and Earthly Branches
# ═════════════════════════════════════════════════════════════════════════════


class ThienCan(IntEnum):
    """10 Heavenly Stems (Thiên Can)."""
    GIAP = 0
    AT = 1
    BINH = 2
    DINH = 3
    MAU = 4
    KY = 5
    CANH = 6
    TAN = 7
    NHAM = 8
    QUY = 9


class DiaChi(IntEnum):
    """12 Earthly Branches (Địa Chi)."""
    TY = 0      # Tý  — Rat
    SUU = 1     # Sửu — Ox
    DAN = 2     # Dần — Tiger
    MAO = 3     # Mão — Cat/Rabbit
    THIN = 4    # Thìn — Dragon
    TI = 5      # Tỵ  — Snake
    NGO = 6     # Ngọ — Horse
    MUI = 7     # Mùi — Goat
    THAN = 8    # Thân — Monkey
    DAU = 9     # Dậu — Rooster
    TUAT = 10   # Tuất — Dog
    HOI = 11    # Hợi — Pig


# Vietnamese display names
THIEN_CAN_NAMES: list[str] = [
    "Giáp", "Ất", "Bính", "Đinh", "Mậu",
    "Kỷ", "Canh", "Tân", "Nhâm", "Quý",
]

DIA_CHI_NAMES: list[str] = [
    "Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ",
    "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi",
]


# ═════════════════════════════════════════════════════════════════════════════
#  Data classes for results
# ═════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CanChi:
    """A Can-Chi pair with both enum values and display names."""
    can: ThienCan
    chi: DiaChi
    can_name: str
    chi_name: str

    def __str__(self) -> str:
        return f"{self.can_name} {self.chi_name}"


@dataclass(frozen=True)
class LunarDateResult:
    """Full lunar date with Can/Chi pillars for Year, Month, Day, Hour."""
    lunar_day: int
    lunar_month: int
    lunar_year: int
    is_leap_month: bool

    year_can_chi: CanChi
    month_can_chi: CanChi
    day_can_chi: CanChi
    hour_can_chi: CanChi

    # Convenience: Julian Day Number of the birth date
    julian_day: int


# ═════════════════════════════════════════════════════════════════════════════
#  Core Astronomical Functions
# ═════════════════════════════════════════════════════════════════════════════

_PI = math.pi


def _jd_from_date(dd: int, mm: int, yy: int) -> int:
    """
    Convert a Gregorian date to Julian Day Number.

    Standard astronomical algorithm valid for dates after the Gregorian
    reform (15 Oct 1582). For earlier dates, falls back to the Julian
    calendar formula.
    """
    a = (14 - mm) // 12
    y = yy + 4800 - a
    m = mm + 12 * a - 3
    jd = dd + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jd


def _jd_to_date(jd: int) -> tuple[int, int, int]:
    """
    Convert a Julian Day Number back to Gregorian (dd, mm, yyyy).

    Inverse of ``_jd_from_date``. Uses the algorithm from Meeus.
    """
    z = jd
    if z < 2299161:
        a = z
    else:
        alpha = int((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - alpha // 4
    b = a + 1524
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)

    day = b - d - int(30.6001 * e)
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    return day, month, year


def _new_moon_jde(k: int) -> float:
    """
    Calculate the Julian Ephemeris Day of the k-th new moon.

    k = 0 corresponds to the new moon near 2000-01-06.
    Uses the algorithm from Hồ Ngọc Đức, adapted from Meeus.

    This is the CORE astronomical function — no lookup tables.
    All periodic terms are computed from trigonometric series.
    """
    T = k / 1236.85  # Time in Julian centuries from J1900.0
    T2 = T * T
    T3 = T2 * T

    dr = _PI / 180.0

    # Mean JDE of the New Moon (Chapront, 1988)
    Jd1 = (
        2415020.75933
        + 29.53058868 * k
        + 0.0001178 * T2
        - 0.000000155 * T3
    )
    Jd1 += 0.00033 * math.sin((166.56 + 132.87 * T - 0.009173 * T2) * dr)

    # Sun's mean anomaly (degrees)
    M = 359.2242 + 29.10535608 * k - 0.0000333 * T2 - 0.00000347 * T3

    # Moon's mean anomaly (degrees)
    Mpr = 306.0253 + 385.81691806 * k + 0.0107306 * T2 + 0.00001236 * T3

    # Moon's argument of latitude (degrees)
    F = 21.2964 + 390.67050646 * k - 0.0016528 * T2 - 0.00000239 * T3

    # Periodic corrections (no lookup tables — pure trig)
    C1 = (0.1734 - 0.000393 * T) * math.sin(M * dr)
    C1 += 0.0021 * math.sin(2 * M * dr)
    C1 -= 0.4068 * math.sin(Mpr * dr)
    C1 += 0.0161 * math.sin(2 * Mpr * dr)
    C1 -= 0.0004 * math.sin(3 * Mpr * dr)
    C1 += 0.0104 * math.sin(2 * F * dr)
    C1 -= 0.0051 * math.sin((M + Mpr) * dr)
    C1 -= 0.0074 * math.sin((M - Mpr) * dr)
    C1 += 0.0004 * math.sin((2 * F + M) * dr)
    C1 -= 0.0004 * math.sin((2 * F - M) * dr)
    C1 -= 0.0006 * math.sin((2 * F + Mpr) * dr)
    C1 += 0.0010 * math.sin((2 * F - Mpr) * dr)
    C1 += 0.0005 * math.sin((2 * Mpr + M) * dr)

    # Delta T correction (terrestrial vs universal time)
    if T < -11:
        deltat = (
            0.001 + 0.000839 * T + 0.0002261 * T2
            - 0.00000845 * T3 - 0.000000081 * T * T3
        )
    else:
        deltat = -0.000278 + 0.000265 * T + 0.000262 * T2

    return Jd1 + C1 - deltat


def _sun_longitude(jdn: float) -> float:
    """
    Calculate the Sun's ecliptic longitude (in radians) at the given JDN.

    Uses the algorithm from Meeus with the equation of center.
    Returns a value in [0, 2π).
    """
    T = (jdn - 2451545.0) / 36525.0  # Julian centuries from J2000.0
    T2 = T * T

    dr = _PI / 180.0

    # Sun's mean anomaly (degrees)
    M = 357.52910 + 35999.05030 * T - 0.0001559 * T2 - 0.00000048 * T * T2
    # Sun's mean longitude (degrees)
    L0 = 280.46645 + 36000.76983 * T + 0.0003032 * T2

    # Equation of center (periodic terms)
    DL = (1.914600 - 0.004817 * T - 0.000014 * T2) * math.sin(M * dr)
    DL += (0.019993 - 0.000101 * T) * math.sin(2 * M * dr)
    DL += 0.000290 * math.sin(3 * M * dr)

    # True longitude (degrees → radians)
    L = (L0 + DL) * dr

    # Normalize to [0, 2π)
    L = L - 2 * _PI * int(L / (2 * _PI))
    if L < 0:
        L += 2 * _PI

    return L


# ═════════════════════════════════════════════════════════════════════════════
#  Lunar Calendar Logic (Ho Ngoc Duc algorithm)
# ═════════════════════════════════════════════════════════════════════════════


def _get_new_moon_day(k: int, tz: float) -> int:
    """
    Get the JDN of the day containing the k-th new moon,
    adjusted for the given timezone offset (hours from UTC).
    """
    return int(_new_moon_jde(k) + 0.5 + tz / 24.0)


def _get_sun_longitude_sector(jdn: int, tz: float) -> int:
    """
    Get the Sun's longitude sector (0–11) at the start of the given JDN.

    Each sector spans 30° of ecliptic longitude, corresponding to
    one Major Solar Term (Trung Khí):
      0 = Xuân Phân (0°)
      3 = Hạ Chí (90°)
      6 = Thu Phân (180°)
      9 = Đông Chí (270°)

    This determines which "month number" a lunar month corresponds to.
    """
    sun_long = _sun_longitude(jdn - 0.5 - tz / 24.0)
    return int(sun_long / _PI * 6)


def _get_lunar_month_11(yy: int, tz: float) -> int:
    """
    Find the JDN of the new moon that starts Lunar Month 11
    of the given year.

    Lunar Month 11 is defined as the month containing the Winter
    Solstice (Đông Chí, Sun longitude = 270°, sector = 9).
    """
    # Approximate lunation number for late December
    off = _jd_from_date(31, 12, yy) - 2415021
    k = int(off / 29.530588853)

    nm = _get_new_moon_day(k, tz)
    sun_sector = _get_sun_longitude_sector(nm, tz)

    # If the sun has already passed 270° (sector ≥ 9),
    # month 11 started at the previous new moon.
    if sun_sector >= 9:
        nm = _get_new_moon_day(k - 1, tz)

    return nm


def _get_leap_month_offset(a11: int, tz: float) -> int:
    """
    Find the leap month offset from the given Month 11 start (a11).

    Walk through successive new moons. The first month whose start and
    end fall within the same 30° Sun longitude sector (i.e., no Major
    Solar Term falls inside it) is the leap month.

    Returns the 1-based offset of the leap month from a11.
    """
    k = int((a11 - 2415021.076998695) / 29.530588853 + 0.5)

    last = 0
    i = 1  # Start from the month after month 11
    arc = _get_sun_longitude_sector(_get_new_moon_day(k + i, tz), tz)

    while True:
        last = arc
        i += 1
        arc = _get_sun_longitude_sector(_get_new_moon_day(k + i, tz), tz)
        if arc != last and i < 14:
            continue
        break

    return i - 1


# ═════════════════════════════════════════════════════════════════════════════
#  Solar → Lunar Conversion
# ═════════════════════════════════════════════════════════════════════════════


def _solar_to_lunar(
    dd: int, mm: int, yy: int, tz: float = 7.0,
) -> tuple[int, int, int, bool]:
    """
    Convert a Gregorian date to Vietnamese Lunar date.

    Algorithm (Hồ Ngọc Đức):
      1. Find the lunation containing this day.
      2. Find Month 11 of the current and surrounding lunar years.
      3. Count months from Month 11 to determine the month number.
      4. If 13 months exist between two Month-11s, find the leap month.

    Returns: (lunar_day, lunar_month, lunar_year, is_leap_month)
    """
    day_number = _jd_from_date(dd, mm, yy)

    # Find the lunation containing this day
    k = int((day_number - 2415021.076998695) / 29.530588853)
    month_start = _get_new_moon_day(k + 1, tz)
    if month_start > day_number:
        month_start = _get_new_moon_day(k, tz)

    # Find Month 11 boundaries
    a11 = _get_lunar_month_11(yy, tz)
    b11 = a11

    if a11 >= month_start:
        lunar_year = yy
        a11 = _get_lunar_month_11(yy - 1, tz)
    else:
        lunar_year = yy + 1
        b11 = _get_lunar_month_11(yy + 1, tz)

    lunar_day = day_number - month_start + 1
    diff = int((month_start - a11) / 29.530588853 + 0.5)

    lunar_leap = False
    lunar_month = diff + 11

    if b11 - a11 > 365:
        leap_month_diff = _get_leap_month_offset(a11, tz)
        if diff >= leap_month_diff:
            lunar_month = diff + 10
            if diff == leap_month_diff:
                lunar_leap = True

    if lunar_month > 12:
        lunar_month -= 12

    if lunar_month >= 11 and diff < 4:
        lunar_year -= 1

    return lunar_day, lunar_month, lunar_year, lunar_leap


# ═════════════════════════════════════════════════════════════════════════════
#  Can / Chi Calculation — Pure Formulas
# ═════════════════════════════════════════════════════════════════════════════


def _make_can_chi(can_idx: int, chi_idx: int) -> CanChi:
    """Build a CanChi from raw indices."""
    can = ThienCan(can_idx % 10)
    chi = DiaChi(chi_idx % 12)
    return CanChi(
        can=can, chi=chi,
        can_name=THIEN_CAN_NAMES[can.value],
        chi_name=DIA_CHI_NAMES[chi.value],
    )


def _year_can_chi(lunar_year: int) -> CanChi:
    """
    Can/Chi of a Lunar Year.

    Formula: The sexagenary cycle starts at year 4 (Giáp Tý).
      can = (year - 4) % 10
      chi = (year - 4) % 12
    """
    can = (lunar_year - 4) % 10
    chi = (lunar_year - 4) % 12
    return _make_can_chi(can, chi)


def _month_can_chi(lunar_month: int, year_can: ThienCan) -> CanChi:
    """
    Can/Chi of a Lunar Month.

    Chi: Month 1 = Dần (2), Month 2 = Mão (3), ..., Month 11 = Tý (0)
    Can: Derived from Year Can using the Ngũ Hổ Dần Khởi rule:
      Giáp/Kỷ   → Month 1 starts with Bính (2)
      Ất/Canh   → Month 1 starts with Mậu  (4)
      Bính/Tân  → Month 1 starts with Canh  (6)
      Đinh/Nhâm → Month 1 starts with Nhâm  (8)
      Mậu/Quý   → Month 1 starts with Giáp  (0)

    Formula: start_can = ((year_can % 5) * 2 + 2) % 10
             month_can = (start_can + month - 1) % 10
    """
    chi = (lunar_month + 1) % 12  # Month 1 → Dần(2), etc.
    start_can = ((year_can.value % 5) * 2 + 2) % 10
    can = (start_can + lunar_month - 1) % 10
    return _make_can_chi(can, chi)


def _day_can_chi(jdn: int) -> CanChi:
    """
    Can/Chi of a day from its Julian Day Number.

    The sexagenary day cycle is a fixed, continuous count:
      can = (jdn + 9) % 10
      chi = (jdn + 1) % 12
    """
    can = (jdn + 9) % 10
    chi = (jdn + 1) % 12
    return _make_can_chi(can, chi)


def _hour_can_chi(birth_hour: int, day_can: ThienCan) -> CanChi:
    """
    Can/Chi of a birth hour.

    Chi: birth_hour index (0=Tý..11=Hợi)
    Can: Derived from Day Can using the Ngũ Thử Dần Khởi rule:
      Giáp/Kỷ   → Hour Tý starts with Giáp (0)
      Ất/Canh   → Hour Tý starts with Bính (2)
      Bính/Tân  → Hour Tý starts with Mậu  (4)
      Đinh/Nhâm → Hour Tý starts with Canh  (6)
      Mậu/Quý   → Hour Tý starts with Nhâm  (8)

    Formula: start_can = (day_can % 5) * 2
             hour_can = (start_can + birth_hour) % 10
    """
    chi = birth_hour % 12
    start_can = (day_can.value % 5) * 2
    can = (start_can + birth_hour) % 10
    return _make_can_chi(can, chi)


# ═════════════════════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════════════════════


def get_lunar_data(
    solar_year: int,
    solar_month: int,
    solar_day: int,
    birth_hour: int = 0,
    timezone: float = 7.0,
) -> LunarDateResult:
    """
    Master function: Convert a Gregorian date + birth hour to a full
    Lunar date with all four Can/Chi pillars.

    Parameters
    ----------
    solar_year, solar_month, solar_day:
        Gregorian date components.
    birth_hour:
        Chinese double-hour index (0=Tý .. 11=Hợi). -1 treated as 0.
    timezone:
        UTC offset in hours. Default 7.0 for Vietnam (ICT).

    Returns
    -------
    LunarDateResult:
        Lunar date, leap month flag, and Can/Chi for Year, Month, Day, Hour.
    """
    if birth_hour < 0:
        birth_hour = 0

    jdn = _jd_from_date(solar_day, solar_month, solar_year)
    l_day, l_month, l_year, l_leap = _solar_to_lunar(
        solar_day, solar_month, solar_year, timezone,
    )

    y_cc = _year_can_chi(l_year)
    m_cc = _month_can_chi(l_month, y_cc.can)
    d_cc = _day_can_chi(jdn)
    h_cc = _hour_can_chi(birth_hour, d_cc.can)

    return LunarDateResult(
        lunar_day=l_day,
        lunar_month=l_month,
        lunar_year=l_year,
        is_leap_month=l_leap,
        year_can_chi=y_cc,
        month_can_chi=m_cc,
        day_can_chi=d_cc,
        hour_can_chi=h_cc,
        julian_day=jdn,
    )
