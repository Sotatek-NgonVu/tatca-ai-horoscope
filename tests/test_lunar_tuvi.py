"""
Tests for the astronomical Lunar Engine and Tu Vi Calculator.

Validates:
  1. Lunar Engine: Gregorian → Lunar conversion accuracy
  2. Lunar Engine: Can/Chi pillar calculations
  3. Tu Vi Calculator: Mệnh/Thân placement
  4. Tu Vi Calculator: Cục computation
  5. Tu Vi Calculator: Full chart generation
  6. Integration: Engine → Calculator → Renderer pipeline
"""

from __future__ import annotations

import pytest

from app.services.lunar_engine import (
    DiaChi,
    ThienCan,
    LunarDateResult,
    get_lunar_data,
    _jd_from_date,
    _jd_to_date,
    _solar_to_lunar,
)
from app.services.tuvi_calculator import (
    House,
    _calc_menh_chi,
    _calc_than_chi,
    _calc_tu_vi_pos,
    _calc_thien_phu_pos,
    _get_cuc,
    _get_cuc_index,
    _calc_nap_am,
    calculate_chart,
    CUNG_NAMES,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Julian Day Conversion Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestJulianDay:
    """Test Gregorian ↔ Julian Day Number conversion."""

    def test_j2000_epoch(self):
        """J2000.0 = 2000-01-01 → JDN 2451545."""
        assert _jd_from_date(1, 1, 2000) == 2451545

    def test_unix_epoch(self):
        """Unix epoch: 1970-01-01 → JDN 2440588."""
        assert _jd_from_date(1, 1, 1970) == 2440588

    def test_known_date_1900(self):
        """1900-01-01 → JDN 2415021."""
        assert _jd_from_date(1, 1, 1900) == 2415021

    def test_roundtrip(self):
        """JDN → Gregorian → JDN should be identity."""
        for jd in [2451545, 2440588, 2415021, 2460000]:
            d, m, y = _jd_to_date(jd)
            assert _jd_from_date(d, m, y) == jd


# ═════════════════════════════════════════════════════════════════════════════
#  Lunar Conversion Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestLunarConversion:
    """Test Solar → Lunar date conversion accuracy."""

    def test_lunar_new_year_2000(self):
        """2000-02-05 is Lunar New Year (1/1/Canh Thìn)."""
        r = get_lunar_data(2000, 2, 5)
        assert r.lunar_day == 1
        assert r.lunar_month == 1
        assert r.lunar_year == 2000
        assert r.is_leap_month is False

    def test_lunar_new_year_2025(self):
        """2025-01-29 is Lunar New Year (1/1/Ất Tỵ)."""
        r = get_lunar_data(2025, 1, 29)
        assert r.lunar_day == 1
        assert r.lunar_month == 1
        assert r.lunar_year == 2025

    def test_1990_01_15(self):
        """1990-01-15 → Lunar 19/12/1989."""
        r = get_lunar_data(1990, 1, 15)
        assert r.lunar_day == 19
        assert r.lunar_month == 12
        assert r.lunar_year == 1989

    def test_leap_month_detection(self):
        """1995-10-20 falls in a leap month (leap month 8, 1995)."""
        r = get_lunar_data(1995, 10, 20)
        assert r.is_leap_month is True
        assert r.lunar_month == 8

    def test_non_leap_month(self):
        """2000-03-15 is not in a leap month."""
        r = get_lunar_data(2000, 3, 15)
        assert r.is_leap_month is False

    def test_end_of_year(self):
        """2024-12-31 should convert without error."""
        r = get_lunar_data(2024, 12, 31)
        assert r.lunar_day > 0
        assert 1 <= r.lunar_month <= 12


# ═════════════════════════════════════════════════════════════════════════════
#  Can / Chi Pillar Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestCanChiPillars:
    """Test the four Can/Chi pillar calculations."""

    def test_year_canh_thin_2000(self):
        """Year 2000 = Canh Thìn."""
        r = get_lunar_data(2000, 6, 1)  # Mid-year, safely in 2000
        assert r.year_can_chi.can == ThienCan.CANH
        assert r.year_can_chi.chi == DiaChi.THIN

    def test_year_ky_ty_1989(self):
        """Year 1989 = Kỷ Tỵ."""
        r = get_lunar_data(1989, 6, 1)
        assert r.year_can_chi.can == ThienCan.KY
        assert r.year_can_chi.chi == DiaChi.TI

    def test_year_at_ty_2025(self):
        """Year 2025 = Ất Tỵ."""
        r = get_lunar_data(2025, 6, 1)
        assert r.year_can_chi.can == ThienCan.AT
        assert r.year_can_chi.chi == DiaChi.TI

    def test_year_can_chi_display(self):
        """Year Can/Chi should have proper Vietnamese names."""
        r = get_lunar_data(2000, 6, 1)
        assert r.year_can_chi.can_name == "Canh"
        assert r.year_can_chi.chi_name == "Thìn"
        assert str(r.year_can_chi) == "Canh Thìn"

    def test_month_can_chi_exists(self):
        """Month Can/Chi should be computed."""
        r = get_lunar_data(2000, 6, 1, birth_hour=0)
        assert r.month_can_chi.can_name in [
            "Giáp", "Ất", "Bính", "Đinh", "Mậu",
            "Kỷ", "Canh", "Tân", "Nhâm", "Quý",
        ]

    def test_hour_can_chi_ty(self):
        """Birth hour 0 should be Tý."""
        r = get_lunar_data(2000, 6, 1, birth_hour=0)
        assert r.hour_can_chi.chi == DiaChi.TY

    def test_hour_can_chi_ngo(self):
        """Birth hour 6 should be Ngọ."""
        r = get_lunar_data(2000, 6, 1, birth_hour=6)
        assert r.hour_can_chi.chi == DiaChi.NGO

    def test_negative_birth_hour_defaults_to_zero(self):
        """Birth hour -1 should default to Tý (0)."""
        r = get_lunar_data(2000, 6, 1, birth_hour=-1)
        assert r.hour_can_chi.chi == DiaChi.TY


# ═════════════════════════════════════════════════════════════════════════════
#  Mệnh / Thân Placement Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestMenhThanPlacement:
    """Test Mệnh and Thân palace placement."""

    def test_menh_month1_hour0(self):
        """Month 1, Hour Tý → Mệnh at Dần."""
        assert _calc_menh_chi(1, 0) == 2  # Dần

    def test_menh_month1_hour6(self):
        """Month 1, Hour Ngọ → Mệnh at Thân."""
        assert _calc_menh_chi(1, 6) == (2 + 0 - 6) % 12  # 8 = Thân

    def test_than_month1_hour0(self):
        """Month 1, Hour Tý → Thân at Dần."""
        assert _calc_than_chi(1, 0) == 2

    def test_menh_than_symmetric(self):
        """Mệnh + Thân should sum to 2*(2 + month - 1) mod 12."""
        for m in range(1, 13):
            for h in range(12):
                menh = _calc_menh_chi(m, h)
                than = _calc_than_chi(m, h)
                assert (menh + than) % 12 == (2 * (2 + m - 1)) % 12


# ═════════════════════════════════════════════════════════════════════════════
#  Cục Calculation Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestCucCalculation:
    """Test Cục determination from Year Can + Mệnh Chi."""

    def test_cuc_giap_ty(self):
        """Giáp (0), Mệnh=Tý (0) → Thủy Nhị Cục (2)."""
        name, val = _get_cuc(0, 0)
        assert val == 2
        assert "Thủy" in name

    def test_cuc_returns_valid_range(self):
        """All Can/Chi combinations should produce a valid Cục (2–6)."""
        for can in range(10):
            for chi in range(12):
                _, val = _get_cuc(can, chi)
                assert val in (2, 3, 4, 5, 6)

    def test_cuc_symmetric_can_pairs(self):
        """Giáp/Kỷ should produce same Cục for same Mệnh Chi."""
        for chi in range(12):
            _, v1 = _get_cuc(0, chi)  # Giáp
            _, v2 = _get_cuc(5, chi)  # Kỷ
            assert v1 == v2


# ═════════════════════════════════════════════════════════════════════════════
#  Tử Vi Star Position Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestTuViPosition:
    """Test Tử Vi and Thiên Phủ star placement."""

    def test_tu_vi_day_equals_cuc(self):
        """When day == cuc, Tử Vi stays at Dần (2)."""
        for cuc in (2, 3, 4, 5, 6):
            pos = _calc_tu_vi_pos(cuc, cuc)
            assert pos == 2  # Dần

    def test_thien_phu_symmetric_at_dan(self):
        """When Tử Vi is at Dần, Thiên Phủ is also at Dần."""
        assert _calc_thien_phu_pos(2) == 2

    def test_thien_phu_mirror(self):
        """Thiên Phủ mirrors Tử Vi across Dần–Thân axis."""
        assert _calc_thien_phu_pos(3) == 1   # Mão → Sửu
        assert _calc_thien_phu_pos(4) == 0   # Thìn → Tý
        assert _calc_thien_phu_pos(5) == 11  # Tỵ → Hợi


# ═════════════════════════════════════════════════════════════════════════════
#  Nạp Âm Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestNapAm:
    """Test Nạp Âm element calculation."""

    def test_giap_ty_is_kim(self):
        """Giáp Tý → Hải Trung Kim → Kim."""
        element, full_name = _calc_nap_am(0, 0)
        assert element == "Kim"
        assert full_name == "Hải Trung Kim"

    def test_returns_valid_element(self):
        """All Can/Chi pairs should return a valid element."""
        valid = {"Kim", "Mộc", "Thủy", "Hỏa", "Thổ"}
        for can in range(10):
            for chi in range(12):
                if can % 2 == chi % 2:  # Same parity only
                    elem, full_name = _calc_nap_am(can, chi)
                    assert elem in valid
                    assert isinstance(full_name, str)


# ═════════════════════════════════════════════════════════════════════════════
#  Full Chart Generation Tests
# ═════════════════════════════════════════════════════════════════════════════


class TestFullChart:
    """Test end-to-end chart generation."""

    @pytest.fixture
    def sample_chart(self):
        """Generate a chart for testing."""
        lunar = get_lunar_data(1990, 5, 15, birth_hour=3)
        return calculate_chart(lunar, "Test User", "Nam")

    def test_chart_has_12_palaces(self, sample_chart):
        """Chart should contain exactly 12 palaces."""
        assert len(sample_chart["la_so"]) == 12

    def test_all_palace_names_present(self, sample_chart):
        """All 12 canonical palace names should be present."""
        for name in CUNG_NAMES:
            assert name in sample_chart["la_so"]

    def test_each_palace_has_dia_chi(self, sample_chart):
        """Each palace should have a Địa Chi assigned."""
        for palace in sample_chart["la_so"].values():
            assert palace["dia_chi"] in [
                "Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ",
                "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi",
            ]

    def test_14_major_stars_placed(self, sample_chart):
        """Exactly 14 major stars should be distributed across palaces."""
        all_major = []
        for palace in sample_chart["la_so"].values():
            all_major.extend(palace.get("chinh_tinh_raw", palace["chinh_tinh"]))
        assert len(all_major) == 14

    def test_tu_vi_star_present(self, sample_chart):
        """Tử Vi star should appear in exactly one palace."""
        count = sum(
            1 for p in sample_chart["la_so"].values()
            if "Tử Vi" in p.get("chinh_tinh_raw", p["chinh_tinh"])
        )
        assert count == 1

    def test_thien_phu_star_present(self, sample_chart):
        """Thiên Phủ star should appear in exactly one palace."""
        count = sum(
            1 for p in sample_chart["la_so"].values()
            if "Thiên Phủ" in p.get("chinh_tinh_raw", p["chinh_tinh"])
        )
        assert count == 1

    def test_owner_info_complete(self, sample_chart):
        """Owner info should contain all required fields."""
        info = sample_chart["thong_tin_chu_nhan"]
        required = ["ten", "gioi_tinh", "am_lich", "menh_cung", "than_cung",
                     "ban_menh", "cuc", "am_duong_thuan_nghich"]
        for field in required:
            assert field in info, f"Missing field: {field}"

    def test_tu_tru_present(self, sample_chart):
        """Tứ Trụ (Four Pillars) should be in the chart."""
        tu_tru = sample_chart["thong_tin_chu_nhan"].get("tu_tru")
        assert tu_tru is not None
        assert "nam" in tu_tru
        assert "thang" in tu_tru
        assert "ngay" in tu_tru
        assert "gio" in tu_tru

    def test_dai_han_present(self, sample_chart):
        """At least some palaces should have Đại Hạn periods."""
        dai_han_count = sum(
            1 for p in sample_chart["la_so"].values()
            if "dai_han" in p
        )
        assert dai_han_count == 12  # All palaces get Đại Hạn

    def test_tuan_triet_markers(self, sample_chart):
        """At least some palaces should have Tuần or Triệt markers."""
        marked = sum(
            1 for p in sample_chart["la_so"].values()
            if p.get("tuan_triet", "")
        )
        assert marked >= 2  # At minimum, 2 Tuần + 2 Triệt

    def test_genders_produce_different_dai_han(self):
        """Male and Female charts should have different Đại Hạn directions."""
        lunar = get_lunar_data(1990, 5, 15, birth_hour=3)
        male_chart = calculate_chart(lunar, "Test", "Nam")
        female_chart = calculate_chart(lunar, "Test", "Nu")

        # Đại Hạn should differ (Thuận vs Nghịch)
        male_dh = [p.get("dai_han") for p in male_chart["la_so"].values()]
        female_dh = [p.get("dai_han") for p in female_chart["la_so"].values()]
        assert male_dh != female_dh


# ═════════════════════════════════════════════════════════════════════════════
#  Integration Test: Engine → Calculator → Renderer
# ═════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Test full pipeline integration."""

    def test_real_engine_produces_chart(self):
        """RealTuViEngine.generate_chart should return valid JSON."""
        from app.domain.models import BirthData, Gender
        from app.services.real_tuvi_engine import RealTuViEngine

        engine = RealTuViEngine()
        birth = BirthData(
            name="Integration Test",
            solar_dob="1985-03-20",
            gender=Gender.FEMALE,
            birth_hour=7,
        )
        chart = engine.generate_chart(birth)
        assert "la_so" in chart
        assert len(chart["la_so"]) == 12

    def test_renderer_produces_png(self):
        """PillowChartRenderer should produce valid PNG bytes."""
        from app.domain.models import BirthData, Gender
        from app.services.real_tuvi_engine import RealTuViEngine
        from app.services.chart_renderer import PillowChartRenderer

        engine = RealTuViEngine()
        renderer = PillowChartRenderer()

        birth = BirthData(
            name="Render Test",
            solar_dob="2000-06-15",
            gender=Gender.MALE,
            birth_hour=5,
        )
        chart = engine.generate_chart(birth)
        img_bytes = renderer.render_chart(chart)

        # Verify PNG magic bytes
        assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(img_bytes) > 10000  # Should be a substantial image

    def test_all_birth_hours_produce_valid_charts(self):
        """Every birth hour (0–11) should produce a valid 12-palace chart."""
        from app.domain.models import BirthData, Gender
        from app.services.real_tuvi_engine import RealTuViEngine

        engine = RealTuViEngine()
        for hour in range(12):
            birth = BirthData(
                name="Hour Test",
                solar_dob="1995-08-25",
                gender=Gender.MALE,
                birth_hour=hour,
            )
            chart = engine.generate_chart(birth)
            assert len(chart["la_so"]) == 12

            # Verify 14 major stars total
            total_major = sum(
                len(p.get("chinh_tinh_raw", p["chinh_tinh"])) for p in chart["la_so"].values()
            )
            assert total_major == 14, f"Hour {hour}: expected 14 major stars, got {total_major}"
