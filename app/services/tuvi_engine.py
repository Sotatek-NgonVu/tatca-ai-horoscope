"""
Mock Tu Vi (Vietnamese Astrology) Engine adapter.

Returns a hard-coded Tu Vi chart JSON structure for development and
testing.  A real implementation would compute Thien Ban, cung, sao, etc.
from the Lunar calendar conversion of the birth data.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.exceptions import TuViEngineError
from app.core.interfaces import TuViEnginePort
from app.domain.models import BirthData

logger = logging.getLogger(__name__)

# ── Sexagenary cycle (Can-Chi) for Vietnamese astrology ──────────────────────
_CAN = [
    "Canh", "Tan", "Nham", "Quy",
    "Giap", "At", "Binh", "Dinh", "Mau", "Ky",
]
_CHI = [
    "Than", "Dau", "Tuat", "Hoi",
    "Ty", "Suu", "Dan", "Mao",
    "Thin", "Ty_", "Ngo", "Mui",
]

_GIO_LABELS = [
    "Ty (23h-1h)", "Suu (1h-3h)", "Dan (3h-5h)", "Mao (5h-7h)",
    "Thin (7h-9h)", "Ty_ (9h-11h)", "Ngo (11h-13h)", "Mui (13h-15h)",
    "Than (15h-17h)", "Dau (17h-19h)", "Tuat (19h-21h)", "Hoi (21h-23h)",
]


def _can_chi(year: int) -> str:
    """Return the Vietnamese can-chi year name for a Gregorian year."""
    return f"{_CAN[year % 10]} {_CHI[year % 12]}"


class MockTuViEngine(TuViEnginePort):
    """
    Mock implementation that returns a realistic-looking Tu Vi chart
    structure.

    This is a placeholder --- swap in the real engine when ready.
    """

    def generate_chart(self, birth_data: BirthData) -> dict[str, Any]:
        """
        Generate a fake Tu Vi chart from the given birth data.

        Returns a dict with keys: menh_cung, than_cung, cuc, major_stars,
        minor_stars, luu_nien, etc.
        """
        try:
            year = int(birth_data.solar_dob.split("-")[0])
            year_can_chi = _can_chi(year)

            gio_label = (
                _GIO_LABELS[birth_data.birth_hour]
                if 0 <= birth_data.birth_hour <= 11
                else "Khong ro"
            )

            chart: dict[str, Any] = {
                "ho_ten": birth_data.name,
                "gioi_tinh": birth_data.gender.value,
                "ngay_sinh_duong": birth_data.solar_dob,
                "gio_sinh": gio_label,
                "nam_can_chi": year_can_chi,
                "menh_cung": "Thien Di",
                "than_cung": "Quan Loc",
                "cuc": "Thuy Nhi Cuc",
                "major_stars": {
                    "Tu Vi": "Menh",
                    "Thien Co": "Huynh De",
                    "Thai Duong": "Phu Mau",
                    "Vu Khuc": "Tai Bach",
                    "Thien Dong": "Phuc Duc",
                    "Liem Trinh": "Quan Loc",
                },
                "minor_stars": {
                    "Van Xuong": "Menh",
                    "Van Khuc": "Tai Bach",
                    "Tham Lang": "Phu The",
                    "Hoa Tinh": "Thien Di",
                },
                "luu_nien_2025": {
                    "cung_luu_nien": "Tai Bach",
                    "luu_hoa_loc": "Vu Khuc",
                    "luu_hoa_quyen": "Thien Dong",
                    "luu_hoa_khoa": "Thai Am",
                    "luu_hoa_ky": "Tham Lang",
                },
                "dai_han": {
                    "cung": "Phu Mau",
                    "range": "2020-2029",
                },
                "tieu_han_2025": {
                    "cung": "Phuc Duc",
                },
            }

            logger.info(
                "Generated mock Tu Vi chart for %s (%s, %s)",
                birth_data.name,
                birth_data.solar_dob,
                year_can_chi,
            )
            return chart

        except Exception as exc:
            raise TuViEngineError(
                f"Failed to generate Tu Vi chart: {exc}", cause=exc
            ) from exc
