"""
Tu Vi Engine — backed by the cloned ``lasotuvi`` library.

Wraps the lasotuvi.App.lapDiaBan() pipeline and converts its output
(diaBan object) to a JSON-serializable dict that fits the existing
TuViEnginePort contract.

Input mapping
─────────────
BirthData.solar_dob   YYYY-MM-DD   → dd / mm / yyyy (int)
BirthData.birth_hour  0-11         → lasotuvi chi index 1-12 (+1 shift)
                                    -1 (unknown) → defaults to 1 (Tý)
BirthData.gender      "Nam"/"Nu"   → 1 / -1
"""

from __future__ import annotations

import logging
import sys
import os
from typing import Any

from app.core.exceptions import TuViEngineError
from app.core.interfaces import TuViEnginePort
from app.domain.models import BirthData

logger = logging.getLogger(__name__)

# ── Make the vendored lasotuvi package importable ────────────────────────────
# lasotuvi/ package lives at project root (app/services/../../lasotuvi)
_PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Lazy imports so startup fails loudly if the folder is missing
try:
    from lasotuvi.App import lapDiaBan           # noqa: E402
    from lasotuvi.DiaBan import diaBan as DiaBanClass  # noqa: E402
    from lasotuvi.ThienBan import lapThienBan    # noqa: E402
    from lasotuvi.AmDuong import diaChi          # noqa: E402

    _LASOTUVI_AVAILABLE = True
except ImportError as _e:
    _LASOTUVI_AVAILABLE = False
    _LASOTUVI_IMPORT_ERROR = _e


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_solar_dob(solar_dob: str) -> tuple[int, int, int]:
    """Parse 'YYYY-MM-DD' → (dd, mm, yyyy) integers."""
    parts = solar_dob.split("-")
    yyyy, mm, dd = int(parts[0]), int(parts[1]), int(parts[2])
    return dd, mm, yyyy


def _birth_hour_to_chi(birth_hour: int) -> int:
    """
    Convert BirthData birth_hour (0=Tý … 11=Hợi, -1=unknown)
    to lasotuvi chi index (1=Tý … 12=Hợi).

    lasotuvi uses 1-based chi: Tý=1, Sửu=2, … Hợi=12.
    BirthData uses 0-based:   Tý=0, Sửu=1, … Hợi=11.
    """
    if birth_hour == -1:
        return 1  # default to Giờ Tý when unknown
    return birth_hour + 1


def _gender_to_int(gender_value: str) -> int:
    """'Nam' → 1, 'Nu' → -1"""
    return 1 if gender_value == "Nam" else -1


def _cung_list_to_dict(dia_ban_obj: Any) -> dict[str, Any]:
    """
    Convert the diaBan.thapNhiCung list (indices 1-12) into a
    human-readable dict keyed by cung name.
    """
    la_so: dict[str, Any] = {}
    for cung in dia_ban_obj.thapNhiCung[1:]:  # skip index-0 placeholder
        ten_cung = getattr(cung, "cungChu", None)
        if not ten_cung:
            continue

        stars_raw: list[dict[str, Any]] = getattr(cung, "cungSao", [])

        chinh_tinh = []
        phu_tinh_tot = []
        phu_tinh_xau = []
        trang_sinh = ""

        for s in stars_raw:
            loai = s.get("saoLoai", 99)
            ten = s.get("saoTen", "")
            dac_tinh = s.get("saoDacTinh", "")
            label = f"{ten} ({dac_tinh})" if dac_tinh else ten

            # Vòng Tràng Sinh — lấy tên ngôi sao đầu tiên trong vòng
            if s.get("vongTrangSinh") == 1 and not trang_sinh:
                trang_sinh = ten

            if loai == 1:
                chinh_tinh.append(label)
            elif loai >= 11:
                phu_tinh_xau.append(label)
            else:
                phu_tinh_tot.append(label)

        # Đại Hạn: lasotuvi lưu tuổi bắt đầu là int → format "N–N+9"
        dai_han_raw = getattr(cung, "cungDaiHan", None)
        if isinstance(dai_han_raw, int):
            dai_han = f"{dai_han_raw}–{dai_han_raw + 9}"
        else:
            dai_han = str(dai_han_raw) if dai_han_raw is not None else ""

        # Tuần / Triệt: bool → string cho renderer
        markers = []
        if getattr(cung, "tuanTrung", False):
            markers.append("Tuần")
        if getattr(cung, "trietLo", False):
            markers.append("Triệt")
        tuan_triet = " / ".join(markers)

        entry: dict[str, Any] = {
            "dia_chi": cung.cungTen,
            "hanh_cung": cung.hanhCung,
            "cung_can": "",
            "chinh_tinh": chinh_tinh,
            "cat_tinh": phu_tinh_tot,
            "sat_tinh": phu_tinh_xau,
            "dai_han": dai_han,
            "tieu_han": getattr(cung, "cungTieuHan", None),
            "trang_sinh": trang_sinh,
            "tuan_triet": tuan_triet,
            "is_than": getattr(cung, "cungThan", False),
        }
        la_so[ten_cung] = entry

    return la_so



# ── Engine ────────────────────────────────────────────────────────────────────

class LasoTuViEngine(TuViEnginePort):
    """
    Real Tu Vi engine powered by the ``lasotuvi`` library.

    Uses the Ho Ngoc Duc astronomical algorithm for solar→lunar
    conversion and the complete star-placement (an sao) logic.
    """

    def __init__(self, time_zone: int = 7) -> None:
        if not _LASOTUVI_AVAILABLE:
            raise RuntimeError(
                f"lasotuvi package could not be imported: {_LASOTUVI_IMPORT_ERROR}"
            )
        self._tz = time_zone

    def generate_chart(self, birth_data: BirthData) -> dict[str, Any]:
        """Generate a full Tu Vi lá số from BirthData."""
        try:
            dd, mm, yyyy = _parse_solar_dob(birth_data.solar_dob)
            gio_sinh = _birth_hour_to_chi(birth_data.birth_hour)
            gioi_tinh = _gender_to_int(birth_data.gender.value)

            # ── Step 1: Build the DiaBan (address board / 12 palaces) ────────
            db = lapDiaBan(
                DiaBanClass,
                dd, mm, yyyy,
                gio_sinh,
                gioi_tinh,
                duongLich=True,
                timeZone=self._tz,
            )

            # ── Step 2: Build the ThienBan (metadata / owner info) ───────────
            tb = lapThienBan(
                dd, mm, yyyy,
                gio_sinh,
                gioi_tinh,
                birth_data.name,
                db,
                duongLich=True,
                timeZone=self._tz,
            )

            # ── Step 3: Serialise ─────────────────────────────────────────────
            la_so = _cung_list_to_dict(db)

            chart: dict[str, Any] = {
                "thong_tin_chu_nhan": {
                    "ten": tb.ten,
                    "gioi_tinh": tb.namNu,
                    "ngay_duong_lich": f"{tb.ngayDuong:02d}/{tb.thangDuong:02d}/{tb.namDuong}",
                    "am_lich": (
                        f"Ngày {tb.ngayAm} tháng {tb.thangAm} năm {tb.namAm}"
                        + (" (nhuận)" if getattr(tb, "thangNhuan", 0) else "")
                    ),
                    "gio_sinh": f"{tb.gioSinh} (chi {gio_sinh})",
                    "menh_cung": diaChi[db.cungMenh]["tenChi"],
                    "than_cung": diaChi[db.cungThan]["tenChi"],
                    "ban_menh": tb.banMenh,
                    "cuc": tb.tenCuc,
                    "sinh_khac_cuc_menh": tb.sinhKhac,
                    "menh_chu": tb.menhChu,
                    "than_chu": tb.thanChu,
                    "am_duong_menh": tb.amDuongMenh,
                    "am_duong_nam_sinh": tb.amDuongNamSinh,
                    "tu_tru": {
                        "nam": f"{tb.canNamTen} {tb.chiNamTen}",
                        "thang": f"{tb.canThangTen} {tb.chiThangTen}",
                        "ngay": f"{tb.canNgayTen} {tb.chiNgayTen}",
                        "gio": tb.gioSinh,
                    },
                },
                "la_so": la_so,
            }

            logger.info(
                "Generated lasotuvi chart for %s (solar %s, gio=%s, gioi_tinh=%s)",
                birth_data.name,
                birth_data.solar_dob,
                gio_sinh,
                birth_data.gender.value,
            )
            return chart

        except Exception as exc:
            raise TuViEngineError(
                f"lasotuvi chart generation failed: {exc}", cause=exc
            ) from exc
