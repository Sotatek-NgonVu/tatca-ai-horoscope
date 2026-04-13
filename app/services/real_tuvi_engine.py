"""
Real Tu Vi (Vietnamese Astrology) Engine — Production Implementation.

Orchestrates Module 1 (Lunar Engine) and Module 2 (Tu Vi Calculator)
behind the ``TuViEnginePort`` interface.

Pipeline: BirthData → Lunar Engine → Tu Vi Calculator → Chart JSON

This module contains NO calculation logic — it delegates entirely
to ``lunar_engine`` and ``tuvi_calculator``.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.interfaces import TuViEnginePort
from app.domain.models import BirthData
from app.services.lunar_engine import get_lunar_data
from app.services.tuvi_calculator import calculate_chart

logger = logging.getLogger(__name__)


class RealTuViEngine(TuViEnginePort):
    """
    Production Tu Vi (Vietnamese Astrology) calculation engine.

    Coordinates the astronomical lunar engine and Tu Vi calculator
    to produce a complete chart from Gregorian birth data.
    """

    def generate_chart(self, birth_data: BirthData) -> dict[str, Any]:
        """
        Generate a complete Tu Vi chart.

        Pipeline:
          1. Parse solar DOB
          2. Convert to Lunar date + Can/Chi pillars (Module 1)
          3. Calculate full chart with all stars (Module 2)
          4. Return structured JSON
        """
        # ── 1. Parse solar DOB ───────────────────────────────────────
        parts = birth_data.solar_dob.split("-")
        s_year, s_month, s_day = int(parts[0]), int(parts[1]), int(parts[2])
        gender_str = birth_data.gender.value  # "Nam" or "Nu"

        # Handle unknown birth hour → default to Tý (0)
        hour = birth_data.birth_hour if birth_data.birth_hour >= 0 else 0

        # ── 2. Lunar conversion (Module 1: Astronomical Engine) ──────
        lunar_data = get_lunar_data(
            solar_year=s_year,
            solar_month=s_month,
            solar_day=s_day,
            birth_hour=hour,
            timezone=7.0,  # Vietnam (ICT)
        )

        logger.info(
            "Lunar conversion: %s → %d/%d/%d %s, Can/Chi=%s",
            birth_data.solar_dob,
            lunar_data.lunar_day, lunar_data.lunar_month, lunar_data.lunar_year,
            "(leap)" if lunar_data.is_leap_month else "",
            lunar_data.year_can_chi,
        )

        # ── 3. Tu Vi calculation (Module 2: Star Placement) ─────────
        chart = calculate_chart(
            lunar_data=lunar_data,
            name=birth_data.name,
            gender=gender_str,
        )

        # Override the solar date field with the actual Gregorian date
        chart["thong_tin_chu_nhan"]["ngay_duong_lich"] = birth_data.solar_dob

        logger.info(
            "Chart generated for %s: Mệnh=%s, Cục=%s",
            birth_data.name,
            chart["thong_tin_chu_nhan"]["menh_cung"],
            chart["thong_tin_chu_nhan"]["cuc"],
        )

        return chart
