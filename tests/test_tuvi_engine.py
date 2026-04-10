"""
Unit tests for app/services/tuvi_engine.py (MockTuViEngine)

Tests the can-chi calculation logic and chart generation behaviour.
No external dependencies — purely local logic.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import TuViEngineError
from app.domain.models import BirthData, Gender
from app.services.tuvi_engine import MockTuViEngine, _can_chi


# =============================================================================
#  _can_chi() — sexagenary cycle helper
# =============================================================================


class TestCanChi:
    """
    The Vietnamese sexagenary cycle repeats every 60 years.
    _CAN has 10 elements (year % 10), _CHI has 12 (year % 12).
    """

    def test_year_1990(self) -> None:
        # 1990 % 10 = 0 -> Canh, 1990 % 12 = 10 -> Ngo
        result = _can_chi(1990)
        assert result == "Canh Ngo"

    def test_year_2000(self) -> None:
        # 2000 % 10 = 0 -> Canh, 2000 % 12 = 8 -> Thin
        result = _can_chi(2000)
        assert result == "Canh Thin"

    def test_year_1985(self) -> None:
        # 1985 % 10 = 5 -> At, 1985 % 12 = 1 -> Dau
        result = _can_chi(1985)
        assert result == "At Suu"

    def test_year_2025(self) -> None:
        # 2025 % 10 = 5 -> At, 2025 % 12 = 9 -> Ty_
        result = _can_chi(2025)
        assert result == "At Ty_"

    def test_returns_string(self) -> None:
        assert isinstance(_can_chi(2000), str)

    def test_contains_space(self) -> None:
        result = _can_chi(1990)
        assert " " in result

    def test_cycle_repeats_every_60_years(self) -> None:
        """The sexagenary cycle repeats every 60 years."""
        assert _can_chi(1900) == _can_chi(1960)
        assert _can_chi(1960) == _can_chi(2020)


# =============================================================================
#  MockTuViEngine.generate_chart()
# =============================================================================


class TestMockTuViEngine:
    def setup_method(self) -> None:
        self.engine = MockTuViEngine()

    def _make_birth_data(
        self,
        name: str = "Test User",
        gender: Gender = Gender.MALE,
        solar_dob: str = "1990-05-15",
        birth_hour: int = 0,
    ) -> BirthData:
        return BirthData(
            name=name,
            gender=gender,
            solar_dob=solar_dob,
            birth_hour=birth_hour,
        )

    def test_returns_dict(self) -> None:
        bd = self._make_birth_data()
        chart = self.engine.generate_chart(bd)
        assert isinstance(chart, dict)

    def test_chart_contains_name(self) -> None:
        bd = self._make_birth_data(name="Nguyen Van A")
        chart = self.engine.generate_chart(bd)
        assert chart["ho_ten"] == "Nguyen Van A"

    def test_chart_contains_gender(self) -> None:
        bd = self._make_birth_data(gender=Gender.FEMALE)
        chart = self.engine.generate_chart(bd)
        assert chart["gioi_tinh"] == "Nu"

    def test_chart_contains_solar_dob(self) -> None:
        bd = self._make_birth_data(solar_dob="1995-12-25")
        chart = self.engine.generate_chart(bd)
        assert chart["ngay_sinh_duong"] == "1995-12-25"

    def test_chart_contains_gio_label_for_known_hour(self) -> None:
        bd = self._make_birth_data(birth_hour=0)
        chart = self.engine.generate_chart(bd)
        # hour 0 = Ty (23h-1h)
        assert "Ty" in chart["gio_sinh"]

    def test_chart_unknown_birth_hour(self) -> None:
        bd = self._make_birth_data(birth_hour=-1)
        chart = self.engine.generate_chart(bd)
        assert chart["gio_sinh"] == "Khong ro"

    def test_chart_has_menh_cung(self) -> None:
        bd = self._make_birth_data()
        chart = self.engine.generate_chart(bd)
        assert "menh_cung" in chart

    def test_chart_has_than_cung(self) -> None:
        bd = self._make_birth_data()
        chart = self.engine.generate_chart(bd)
        assert "than_cung" in chart

    def test_chart_has_major_stars(self) -> None:
        bd = self._make_birth_data()
        chart = self.engine.generate_chart(bd)
        assert "major_stars" in chart
        assert isinstance(chart["major_stars"], dict)

    def test_chart_has_minor_stars(self) -> None:
        bd = self._make_birth_data()
        chart = self.engine.generate_chart(bd)
        assert "minor_stars" in chart

    def test_chart_has_nam_can_chi(self) -> None:
        bd = self._make_birth_data(solar_dob="1990-01-01")
        chart = self.engine.generate_chart(bd)
        assert "nam_can_chi" in chart
        # 1990 should be Canh Ngo
        assert chart["nam_can_chi"] == "Canh Ngo"

    def test_chart_for_all_valid_birth_hours(self) -> None:
        for hour in range(12):
            bd = self._make_birth_data(birth_hour=hour)
            chart = self.engine.generate_chart(bd)
            assert "gio_sinh" in chart
            assert chart["gio_sinh"] != "Khong ro"

    def test_chart_female_gender(self) -> None:
        bd = self._make_birth_data(gender=Gender.FEMALE, name="Nguyen Thi B")
        chart = self.engine.generate_chart(bd)
        assert chart["gioi_tinh"] == "Nu"
        assert chart["ho_ten"] == "Nguyen Thi B"

    def test_implements_tuvi_engine_port(self) -> None:
        from app.core.interfaces import TuViEnginePort
        assert isinstance(self.engine, TuViEnginePort)
