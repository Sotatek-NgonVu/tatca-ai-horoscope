"""
Tử Vi Đẩu Số — Complete Star Placement Calculator.

Implements the full Tu Vi algorithm per classical rules:
  Bước 1: Ngũ Hổ Độn → Nạp Can cho 12 cung
  Bước 2: An Mệnh/Thân + 12 cung chức năng
  Bước 3: Tính Cục số (Nạp Âm of Cung Mệnh Can-Chi)
  Bước 4: An 14 Chính tinh (Tử Vi ring + Thiên Phủ ring)
  Bước 5: Vòng Thái Tuế, Lộc Tồn/Bác Sĩ, Trường Sinh
  Bước 6: An Phụ tinh (Can/Chi/Tháng/Ngày/Giờ năm)
  Bước 7: Tuần/Triệt, Đại Hạn

Author: Tu Vi Astrology Engine — tatca.ai
"""
from __future__ import annotations
import logging
from typing import Any
from app.services.lunar_engine import (
    DiaChi, LunarDateResult, ThienCan, DIA_CHI_NAMES, THIEN_CAN_NAMES,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════
CUNG_NAMES: list[str] = [
    "Mệnh", "Phụ Mẫu", "Phúc Đức", "Điền Trạch",
    "Quan Lộc", "Nô Bộc", "Thiên Di", "Tật Ách",
    "Tài Bạch", "Tử Tức", "Phu Thê", "Huynh Đệ",
]
NGU_HANH_CUC: list[tuple[str, int]] = [
    ("Thủy Nhị Cục", 2), ("Mộc Tam Cục", 3), ("Kim Tứ Cục", 4),
    ("Thổ Ngũ Cục", 5), ("Hỏa Lục Cục", 6),
]
TRANG_SINH_NAMES: list[str] = [
    "Trường Sinh", "Mộc Dục", "Quan Đới", "Lâm Quan",
    "Đế Vượng", "Suy", "Bệnh", "Tử",
    "Mộ", "Tuyệt", "Thai", "Dưỡng",
]
NAP_AM_FULL: list[str] = [
    "Hải Trung Kim","Lô Trung Hỏa","Đại Lâm Mộc","Lộ Bàng Thổ",
    "Kiếm Phong Kim","Sơn Đầu Hỏa","Giản Hạ Thủy","Thành Đầu Thổ",
    "Bạch Lạp Kim","Dương Liễu Mộc","Tuyền Trung Thủy","Ốc Thượng Thổ",
    "Tích Lịch Hỏa","Tùng Bách Mộc","Trường Lưu Thủy","Sa Trung Kim",
    "Sơn Hạ Hỏa","Bình Địa Mộc","Bích Thượng Thổ","Kim Bạc Kim",
    "Phúc Đăng Hỏa","Thiên Hà Thủy","Đại Dịch Thổ","Thoa Xuyến Kim",
    "Tang Đố Mộc","Đại Khê Thủy","Sa Trung Thổ","Thiên Thượng Hỏa",
    "Thạch Lựu Mộc","Đại Hải Thủy",
]
_NAP_AM_EL = ["Kim","Hỏa","Mộc","Thổ","Thủy"]
_NAP_AM_30: list[int] = [
    0,1,2,3,0,1, 4,3,0,2,4,3, 1,2,4,0,1,2, 3,0,1,4,3,0, 2,4,3,1,2,4,
]
# Miếu/Vượng/Đắc/Bình/Hãm brightness per major star per chi
_BRIGHT: dict[str,dict[int,str]] = {
    "Tử Vi":{6:"M",5:"M",0:"V",1:"V",2:"Đ",10:"Đ"},
    "Thiên Cơ":{0:"M",7:"M",3:"V",9:"V",2:"Đ",6:"H",8:"H"},
    "Thái Dương":{3:"M",5:"M",6:"V",2:"V",4:"Đ",10:"H",11:"H",0:"H"},
    "Vũ Khúc":{4:"M",10:"M",1:"V",7:"V",8:"Đ",9:"Đ"},
    "Thiên Đồng":{0:"M",5:"M",3:"V",11:"V",9:"Đ",6:"H",2:"H"},
    "Liêm Trinh":{2:"M",8:"M",7:"V",11:"Đ",6:"H",5:"H",9:"H"},
    "Thiên Phủ":{10:"M",1:"M",4:"V",7:"V",0:"Đ",6:"Đ"},
    "Thái Âm":{9:"M",11:"M",0:"V",10:"V",1:"Đ",3:"H",6:"H"},
    "Tham Lang":{0:"M",4:"M",10:"V",2:"V",8:"Đ",5:"H",11:"H"},
    "Cự Môn":{0:"M",3:"M",6:"V",11:"V",9:"Đ",4:"H",10:"H"},
    "Thiên Tướng":{0:"M",7:"M",3:"V",9:"V",2:"Đ",6:"H"},
    "Thiên Lương":{0:"M",6:"M",3:"V",1:"Đ",7:"Đ",5:"H",11:"H"},
    "Thất Sát":{2:"M",8:"M",0:"V",6:"V",4:"Đ",10:"Đ"},
    "Phá Quân":{0:"M",6:"M",2:"V",8:"V",4:"Đ",10:"Đ",1:"H",7:"H"},
}
MAJOR_STAR_NAMES: list[str] = list(_BRIGHT.keys())

# ═══════════════════════════════════════════════════════════════════
#  House
# ═══════════════════════════════════════════════════════════════════
class House:
    __slots__ = (
        "name","dia_chi","dia_chi_idx","cung_can",
        "major_stars","minor_lucky","minor_malefic",
        "tuan","triet","dai_han_start","trang_sinh",
    )
    def __init__(self, name: str, idx: int) -> None:
        self.name = name; self.dia_chi_idx = idx
        self.dia_chi = DIA_CHI_NAMES[idx]; self.cung_can = ""
        self.major_stars: list[str] = []; self.minor_lucky: list[str] = []
        self.minor_malefic: list[str] = []; self.tuan = False; self.triet = False
        self.dai_han_start: int|None = None; self.trang_sinh = ""
    def to_dict(self) -> dict[str,Any]:
        markers = []
        if self.tuan: markers.append("Tuần")
        if self.triet: markers.append("Triệt")
        bs = []
        for s in self.major_stars:
            b = _BRIGHT.get(s,{}).get(self.dia_chi_idx,"")
            bs.append(f"{s} ({b})" if b else s)
        r: dict[str,Any] = {
            "dia_chi":self.dia_chi,"cung_can":self.cung_can,
            "chinh_tinh":bs,"chinh_tinh_raw":self.major_stars[:],
            "cat_tinh":self.minor_lucky[:],"sat_tinh":self.minor_malefic[:],
            "tuan_triet":" / ".join(markers),"trang_sinh":self.trang_sinh,
        }
        if self.dai_han_start is not None:
            r["dai_han"] = f"{self.dai_han_start}–{self.dai_han_start+9}"
        return r

# ═══════════════════════════════════════════════════════════════════
#  Bước 1: Ngũ Hổ Độn — Nạp Can cho 12 cung
# ═══════════════════════════════════════════════════════════════════
def _ngu_ho_don(houses: list[House], can_nam: int) -> None:
    """Giáp/Kỷ→Bính Dần, Ất/Canh→Mậu Dần, Bính/Tân→Canh Dần,
       Đinh/Nhâm→Nhâm Dần, Mậu/Quý→Giáp Dần. Thuận nạp 11 cung."""
    start = ((can_nam % 5) * 2 + 2) % 10
    for i in range(12):
        houses[(2 + i) % 12].cung_can = THIEN_CAN_NAMES[(start + i) % 10]

# ═══════════════════════════════════════════════════════════════════
#  Bước 2: An Mệnh / Thân / 12 cung chức năng
# ═══════════════════════════════════════════════════════════════════
def _an_menh(thang: int, gio: int) -> int:
    """Tháng giêng khởi Dần, đếm thuận đến tháng sinh;
       giờ Tý tại vị trí đó, đếm nghịch đến giờ sinh."""
    return (2 + (thang - 1) - gio) % 12

def _an_than(thang: int, gio: int) -> int:
    """Tháng giêng khởi Dần, đếm thuận; giờ đếm thuận."""
    return (2 + (thang - 1) + gio) % 12

def _an_12_cung(houses: list[House], menh_chi: int) -> None:
    """Từ Mệnh, lần lượt an 12 cung đếm thuận theo chi."""
    for i, name in enumerate(CUNG_NAMES):
        houses[(menh_chi + i) % 12].name = name

# ═══════════════════════════════════════════════════════════════════
#  Bước 3: Cục số — Nạp Âm của Can-Chi cung Mệnh
# ═══════════════════════════════════════════════════════════════════
def _nap_am(can: int, chi: int) -> tuple[int, str, str]:
    """Returns (cuc_element_idx, element_name, full_name)."""
    k = (6 * can - 5 * chi) % 60
    pi = k // 2
    ei = _NAP_AM_30[pi]
    return ei, _NAP_AM_EL[ei], NAP_AM_FULL[pi]

def _calc_nap_am(can: int, chi: int) -> tuple[str, str]:
    """Public API: returns (element, full_name)."""
    _, el, fn = _nap_am(can, chi)
    return el, fn

def _tinh_cuc(can_nam: int, menh_chi: int, houses: list[House]) -> tuple[str, int, int]:
    """Cục = Nạp Âm ngũ hành of Can-Chi cung Mệnh.
    Returns (cuc_name, cuc_val, cuc_idx)."""
    # Can of cung Mệnh from Ngũ Hổ Độn
    can_menh_name = houses[menh_chi].cung_can
    can_menh = THIEN_CAN_NAMES.index(can_menh_name)
    ei, _, _ = _nap_am(can_menh, menh_chi)
    # Map element to Cục: Kim→4, Thủy→2, Hỏa→6, Mộc→3, Thổ→5
    _EL_TO_CUC = {0: 2, 1: 4, 2: 1, 3: 3, 4: 0}  # el_idx → cuc_idx
    ci = _EL_TO_CUC[ei]
    return NGU_HANH_CUC[ci][0], NGU_HANH_CUC[ci][1], ci

# ═══════════════════════════════════════════════════════════════════
#  Bước 4: An 14 Chính tinh
# ═══════════════════════════════════════════════════════════════════
def _tim_tu_vi(ngay: int, cuc: int) -> int:
    """Ngày/Cục → vị trí Tử Vi. Lẻ lùi, chẵn tiến."""
    q, r = divmod(ngay, cuc)
    pos = (2 + q) % 12  # Dần + quotient
    if r == 0:
        pos = (pos - 1) % 12  # exact division: back 1 since Dần=position1
        return pos
    du = cuc - r  # amount borrowed
    if du % 2 == 1:  # lẻ → lùi
        return (pos - du) % 12
    else:  # chẵn → tiến
        return (pos + du) % 12

def _tim_thien_phu(tv: int) -> int:
    """Thiên Phủ đối xứng Tử Vi qua trục Dần-Thân."""
    return (4 - tv + 12) % 12

def _an_vong_tu_vi(houses: list[House], tv: int) -> None:
    """Vòng Tử Vi đi thuận: TV, bỏ 3→Liêm Trinh, bỏ 2→Thiên Đồng,
    sát→Vũ Khúc, sát→Thái Dương, bỏ 1→Thiên Cơ."""
    offsets = {"Tử Vi":0,"Liêm Trinh":4,"Thiên Đồng":7,
               "Vũ Khúc":8,"Thái Dương":9,"Thiên Cơ":11}
    for star, off in offsets.items():
        houses[(tv + off) % 12].major_stars.append(star)

def _an_vong_thien_phu(houses: list[House], tp: int) -> None:
    """Vòng Thiên Phủ đi thuận: TP, Thái Âm, Tham Lang, Cự Môn,
    Thiên Tướng, Thiên Lương, Thất Sát, bỏ 3→Phá Quân."""
    stars = ["Thiên Phủ","Thái Âm","Tham Lang","Cự Môn",
             "Thiên Tướng","Thiên Lương","Thất Sát"]
    for i, s in enumerate(stars):
        houses[(tp + i) % 12].major_stars.append(s)
    houses[(tp + 10) % 12].major_stars.append("Phá Quân")  # skip 3

# ═══════════════════════════════════════════════════════════════════
#  Bước 5a: Vòng Thái Tuế (Chi năm)
# ═══════════════════════════════════════════════════════════════════
_THAI_TUE_STARS = [
    "Thái Tuế","Thiếu Dương","Tang Môn","Thiếu Âm",
    "Quan Phù","Tử Phù","Tuế Phá","Long Đức",
    "Bạch Hổ","Phúc Đức","Điếu Khách","Trực Phù",
]
def _an_vong_thai_tue(houses: list[House], chi_nam: int) -> None:
    for i, s in enumerate(_THAI_TUE_STARS):
        houses[(chi_nam + i) % 12].minor_lucky.append(s)

# ═══════════════════════════════════════════════════════════════════
#  Bước 5b: Vòng Lộc Tồn + Bác Sĩ (Can năm)
# ═══════════════════════════════════════════════════════════════════
_LOC_TON_POS = [2,3,5,6,5,6,8,9,11,0]  # Giáp→Dần...bỏ Thìn/Tuất/Sửu/Mùi

_BAC_SI_STARS = [
    "Bác Sĩ","Lực Sĩ","Thanh Long","Tiểu Hao",
    "Tướng Quân","Tấu Thư","Phi Liêm","Hỷ Thần",
    "Bệnh Phù","Đại Hao","Phục Binh","Quan Phủ",
]

def _an_vong_loc_ton(houses: list[House], can_nam: int, is_thuan: bool) -> int:
    """An Lộc Tồn + Bác Sĩ ring. Returns Lộc Tồn position."""
    lt = _LOC_TON_POS[can_nam]
    houses[lt].minor_lucky.append("Lộc Tồn")
    d = 1 if is_thuan else -1
    for i, s in enumerate(_BAC_SI_STARS):
        houses[(lt + d * i) % 12].minor_lucky.append(s)
    return lt

# ═══════════════════════════════════════════════════════════════════
#  Bước 5c: Kình Dương / Đà La (hệ phái Thiên Lương)
# ═══════════════════════════════════════════════════════════════════
def _an_kinh_da(houses: list[House], can_nam: int) -> None:
    """Kình Dương đứng cùng Lực Sĩ (Lộc Tồn+1 thuận),
       Đà La đứng sau Lộc Tồn (Lộc Tồn-1)."""
    lt = _LOC_TON_POS[can_nam]
    houses[(lt + 1) % 12].minor_malefic.append("Kình Dương")
    houses[(lt - 1) % 12].minor_malefic.append("Đà La")

# ═══════════════════════════════════════════════════════════════════
#  Bước 5d: Vòng Trường Sinh (Cục + giới tính)
# ═══════════════════════════════════════════════════════════════════
# Cục element → starting chi. Thủy/Thổ→Thân, Hỏa→Dần, Kim→Tỵ, Mộc→Hợi
_TS_START = {0: 8, 1: 5, 2: 11, 3: 8, 4: 2}  # cuc_idx→chi

def _an_truong_sinh(houses: list[House], cuc_idx: int, is_thuan: bool) -> None:
    start = _TS_START[cuc_idx]
    d = 1 if is_thuan else -1
    for i, s in enumerate(TRANG_SINH_NAMES):
        houses[(start + d * i) % 12].trang_sinh = s

# ═══════════════════════════════════════════════════════════════════
#  Bước 6a: Phụ tinh theo Can năm
# ═══════════════════════════════════════════════════════════════════
def _an_sao_can_nam(houses: list[House], can: int) -> None:
    # Thiên Khôi / Thiên Việt (Giáp Mậu→Sửu/Mùi...)
    _KHOI=[1,0,11,11,1,0,6,7,3,3]
    _VIET=[7,8,9,9,7,8,2,3,5,5]
    houses[_KHOI[can]].minor_lucky.append("Thiên Khôi")
    houses[_VIET[can]].minor_lucky.append("Thiên Việt")
    # Thiên Quan / Thiên Phúc
    _QUAN=[7,4,0,3,6,9,11,10,6,2]
    _PHUC=[9,8,0,11,3,2,6,5,9,8]
    houses[_QUAN[can]].minor_lucky.append("Thiên Quan")
    houses[_PHUC[can]].minor_lucky.append("Thiên Phúc")
    # Lưu Hà: Giáp→Dậu, Ất→Tuất, Bính→Mùi, Đinh→Thân, Mậu→Tỵ...
    _LH=[9,10,7,8,5,6,3,4,1,2]
    houses[_LH[can]].minor_malefic.append("Lưu Hà")
    # Thiên Trù
    _TRU=[5,6,7,8,9,10,11,0,1,2]
    houses[_TRU[can]].minor_lucky.append("Thiên Trù")
    # Quốc Ấn (Lộc Tồn+2 thuận)
    lt = _LOC_TON_POS[can]
    houses[(lt + 2) % 12].minor_lucky.append("Quốc Ấn")
    # Đường Phù (Lộc Tồn+2 nghịch)
    houses[(lt - 2) % 12].minor_lucky.append("Đường Phù")
    # Tứ Hóa
    _TH = [
        ["Liêm Trinh","Phá Quân","Vũ Khúc","Thái Dương"],
        ["Thiên Cơ","Thiên Lương","Tử Vi","Thái Âm"],
        ["Thiên Đồng","Thiên Cơ","Văn Xương","Liêm Trinh"],
        ["Thái Âm","Thiên Đồng","Thiên Cơ","Cự Môn"],
        ["Tham Lang","Thái Âm","Hữu Bật","Thiên Cơ"],
        ["Vũ Khúc","Tham Lang","Thiên Lương","Văn Khúc"],
        ["Thái Dương","Vũ Khúc","Thái Âm","Thiên Đồng"],
        ["Cự Môn","Thái Dương","Văn Khúc","Văn Xương"],
        ["Thiên Lương","Tử Vi","Tả Phụ","Vũ Khúc"],
        ["Phá Quân","Cự Môn","Thái Âm","Tham Lang"],
    ]
    hnames = ["Hóa Lộc","Hóa Quyền","Hóa Khoa","Hóa Kỵ"]
    for hn, ts in zip(hnames, _TH[can]):
        for h in houses:
            if ts in h.major_stars or ts in h.minor_lucky:
                (h.minor_malefic if hn=="Hóa Kỵ" else h.minor_lucky).append(hn)
                break

# ═══════════════════════════════════════════════════════════════════
#  Bước 6b: Phụ tinh theo Chi năm
# ═══════════════════════════════════════════════════════════════════
def _an_sao_chi_nam(houses: list[House], chi: int, menh_chi: int, than_chi: int) -> None:
    # Thiên Không: đi cùng Thiếu Dương (+1 from chi)
    houses[(chi + 1) % 12].minor_malefic.append("Thiên Không")
    # Thiên Mã: xung chiếu chi đầu tam hợp
    _MA={2:8,6:8,10:8, 8:2,0:2,4:2, 5:11,9:11,1:11, 11:5,3:5,7:5}
    houses[_MA[chi]].minor_lucky.append("Thiên Mã")
    # Đào Hoa
    _DH={0:9,4:9,8:9, 3:0,7:0,11:0, 2:3,6:3,10:3, 1:6,5:6,9:6}
    houses[_DH[chi]].minor_lucky.append("Đào Hoa")
    # Hồng Loan / Thiên Hỷ
    hl=(3-chi)%12; houses[hl].minor_lucky.append("Hồng Loan")
    houses[(hl+6)%12].minor_lucky.append("Thiên Hỷ")
    # Long Trì / Phượng Các
    houses[(4+chi)%12].minor_lucky.append("Long Trì")
    houses[(10-chi)%12].minor_lucky.append("Phượng Các")
    # Cô Thần / Quả Tú
    _CT={2:5,3:5,4:5, 5:8,6:8,7:8, 8:11,9:11,10:11, 11:2,0:2,1:2}
    _QT={2:1,3:1,4:1, 5:4,6:4,7:4, 8:7,9:7,10:7, 11:10,0:10,1:10}
    houses[_CT[chi]].minor_malefic.append("Cô Thần")
    houses[_QT[chi]].minor_lucky.append("Quả Tú")
    # Hoa Cái / Kiếp Sát
    houses[(_DH[chi]+6)%12].minor_lucky.append("Hoa Cái")
    _KS={8:5,0:5,4:5, 11:8,3:8,7:8, 2:11,6:11,10:11, 5:2,9:2,1:2}
    houses[_KS[chi]].minor_malefic.append("Kiếp Sát")
    # Phá Toái
    _PT={0:5,6:5, 1:1,7:7, 2:11,8:11, 3:3,9:9, 4:5,10:5, 5:9,11:9}
    houses[_PT[chi]].minor_malefic.append("Phá Toái")
    # Thiên Đức / Nguyệt Đức
    houses[(chi+9)%12].minor_lucky.append("Thiên Đức")
    houses[(chi+4)%12].minor_lucky.append("Nguyệt Đức")
    # Đẩu Quân (tháng giêng=Mão rồi nghịch theo chi)
    houses[(3-chi)%12].minor_lucky.append("Đẩu Quân")
    # Thiên Tài (Mệnh đếm thuận theo chi năm)
    houses[(menh_chi+chi)%12].minor_lucky.append("Thiên Tài")
    # Thiên Thọ (Thân đếm thuận theo chi năm)
    houses[(than_chi+chi)%12].minor_lucky.append("Thiên Thọ")

# ═══════════════════════════════════════════════════════════════════
#  Bước 6c: Phụ tinh theo Tháng sinh
# ═══════════════════════════════════════════════════════════════════
def _an_sao_thang(houses: list[House], m: int) -> None:
    mi = m - 1
    houses[(4+mi)%12].minor_lucky.append("Tả Phụ")
    houses[(10-mi)%12].minor_lucky.append("Hữu Bật")
    houses[(9+mi)%12].minor_malefic.append("Thiên Hình")
    # Thiên Y: Sửu+tháng
    houses[(1+mi)%12].minor_lucky.append("Thiên Y")
    # Thiên Riêu: Dần+tháng
    _RIÊU=[4,5,6,4,5,6,4,5,6,4,5,6]  # traditional month mapping
    houses[_RIÊU[mi]].minor_malefic.append("Thiên Riêu")
    # Thiên Giải / Địa Giải
    houses[(10+mi)%12].minor_lucky.append("Thiên Giải")
    houses[(6+mi)%12].minor_lucky.append("Địa Giải")

# ═══════════════════════════════════════════════════════════════════
#  Bước 6d: Phụ tinh theo Ngày sinh
# ═══════════════════════════════════════════════════════════════════
def _an_sao_ngay(houses: list[House], d: int) -> None:
    di = d - 1
    houses[(4+di)%12].minor_lucky.append("Tam Thai")
    houses[(10-di)%12].minor_lucky.append("Bát Tọa")
    houses[(6+di)%12].minor_lucky.append("Ân Quang")
    houses[(8-di)%12].minor_lucky.append("Thiên Quý")

# ═══════════════════════════════════════════════════════════════════
#  Bước 6e: Phụ tinh theo Giờ sinh
# ═══════════════════════════════════════════════════════════════════
def _an_sao_gio(houses: list[House], gio: int, chi_nam: int, is_thuan: bool) -> None:
    # Văn Xương: Tuất đếm nghịch; Văn Khúc: Thìn đếm thuận
    houses[(10-gio)%12].minor_lucky.append("Văn Xương")
    houses[(4+gio)%12].minor_lucky.append("Văn Khúc")
    # Địa Không / Địa Kiếp: khởi Hợi
    houses[(11-gio)%12].minor_malefic.append("Địa Không")
    houses[(11+gio)%12].minor_malefic.append("Địa Kiếp")
    # Thai Phụ / Phong Cáo
    houses[(6+gio)%12].minor_lucky.append("Thai Phụ")
    houses[(2+gio)%12].minor_lucky.append("Phong Cáo")
    # Hỏa Tinh / Linh Tinh (tam hợp tuổi + giới tính)
    _HOA={2:1,6:1,10:1, 8:2,0:2,4:2, 5:3,9:3,1:3, 11:9,3:9,7:9}
    _LINH={2:3,6:3,10:3, 8:10,0:10,4:10, 5:10,9:10,1:10, 11:10,3:10,7:10}
    hb,lb = _HOA[chi_nam], _LINH[chi_nam]
    d = 1 if is_thuan else -1
    houses[(hb + d*gio)%12].minor_malefic.append("Hỏa Tinh")
    houses[(lb + d*gio)%12].minor_malefic.append("Linh Tinh")
    # Thiên La / Địa Võng (cố định)
    houses[4].minor_malefic.append("Thiên La")
    houses[10].minor_malefic.append("Địa Võng")

# ═══════════════════════════════════════════════════════════════════
#  Tuần / Triệt
# ═══════════════════════════════════════════════════════════════════
def _an_tuan_triet(houses: list[House], can: int, chi: int) -> None:
    s = (chi - can) % 12
    houses[(s+10)%12].tuan = True; houses[(s+11)%12].tuan = True
    g = can % 5
    b = (4 - g) * 2
    houses[b%12].triet = True; houses[(b+1)%12].triet = True

# ═══════════════════════════════════════════════════════════════════
#  Đại Hạn
# ═══════════════════════════════════════════════════════════════════
def _an_dai_han(houses: list[House], menh_chi: int, cuc_val: int, is_thuan: bool) -> None:
    d = 1 if is_thuan else -1
    for i in range(12):
        houses[(menh_chi+d*i)%12].dai_han_start = cuc_val + 10*i

# ═══════════════════════════════════════════════════════════════════
#  Mệnh Chủ / Thân Chủ
# ═══════════════════════════════════════════════════════════════════
_MENH_CHU_7 = ["Tham Lang","Cự Môn","Lộc Tồn","Văn Khúc","Liêm Trinh","Vũ Khúc","Phá Quân"]
_THAN_CHU_6 = ["Hỏa Tinh","Thiên Tướng","Thiên Lương","Thiên Đồng","Văn Xương","Thiên Cơ"]

def _menh_chu(menh_chi: int) -> str:
    return _MENH_CHU_7[menh_chi % 7]

def _than_chu(chi_nam: int) -> str:
    return _THAN_CHU_6[chi_nam % 6]

# ═══════════════════════════════════════════════════════════════════
#  Ngũ Hành tương khắc check
# ═══════════════════════════════════════════════════════════════════
def _check_khac(cuc_el: str, menh_el: str) -> str:
    khac = {"Kim":"Mộc","Mộc":"Thổ","Thổ":"Thủy","Thủy":"Hỏa","Hỏa":"Kim"}
    if khac.get(cuc_el)==menh_el: return f"Cục khắc Bản Mệnh"
    if khac.get(menh_el)==cuc_el: return f"Bản Mệnh khắc Cục"
    return ""

# ═══════════════════════════════════════════════════════════════════
#  MASTER: Generate_TuVi_Chart
# ═══════════════════════════════════════════════════════════════════
def calculate_chart(lunar_data: LunarDateResult, name: str, gender: str) -> dict[str,Any]:
    can = lunar_data.year_can_chi.can.value
    chi = lunar_data.year_can_chi.chi.value
    m = lunar_data.lunar_month
    d = lunar_data.lunar_day
    h = lunar_data.hour_can_chi.chi.value
    is_duong = can % 2 == 0
    is_male = gender == "Nam"
    is_thuan = (is_duong and is_male) or (not is_duong and not is_male)

    # B1: Tạo 12 cung + Nạp Can
    houses: list[House] = [House("", i) for i in range(12)]
    _ngu_ho_don(houses, can)
    # B2: An Mệnh/Thân + 12 cung
    mc = _an_menh(m, h)
    tc = _an_than(m, h)
    _an_12_cung(houses, mc)
    # B3: Cục
    cuc_name, cuc_val, cuc_idx = _tinh_cuc(can, mc, houses)
    # B4: 14 Chính tinh
    tv = _tim_tu_vi(d, cuc_val)
    tp = _tim_thien_phu(tv)
    _an_vong_tu_vi(houses, tv)
    _an_vong_thien_phu(houses, tp)
    # B5: 3 vòng cốt lõi
    _an_vong_thai_tue(houses, chi)
    lt = _an_vong_loc_ton(houses, can, is_thuan)
    _an_kinh_da(houses, can)
    _an_truong_sinh(houses, cuc_idx, is_thuan)
    # B6: Phụ tinh
    _an_sao_can_nam(houses, can)
    _an_sao_chi_nam(houses, chi, mc, tc)
    _an_sao_thang(houses, m)
    _an_sao_ngay(houses, d)
    _an_sao_gio(houses, h, chi, is_thuan)
    # B7: Tuần/Triệt + Đại Hạn
    _an_tuan_triet(houses, can, chi)
    _an_dai_han(houses, mc, cuc_val, is_thuan)

    # Nạp Âm năm sinh
    menh_el, menh_full = _calc_nap_am(can, chi)
    cuc_el = cuc_name.split()[0]
    khac = _check_khac(cuc_el, menh_el)
    am_duong = ("Dương Nam — Thuận" if is_thuan else "Âm Nam — Nghịch") if is_male \
        else ("Âm Nữ — Thuận" if is_thuan else "Dương Nữ — Nghịch")
    am_lich = f"Ngày {d} tháng {m} năm {THIEN_CAN_NAMES[can]} {DIA_CHI_NAMES[chi]} ({lunar_data.lunar_year})"
    if lunar_data.is_leap_month: am_lich += " (nhuận)"

    la_so: dict[str,dict[str,Any]] = {}
    for pn in CUNG_NAMES:
        for hx in houses:
            if hx.name == pn:
                dd = hx.to_dict()
                if hx.dia_chi_idx == tc: dd["is_than"] = True
                la_so[pn] = dd; break

    return {
        "thong_tin_chu_nhan": {
            "ten": name, "gioi_tinh": gender,
            "ngay_duong_lich": str(lunar_data.julian_day),
            "am_lich": am_lich,
            "gio_sinh": f"{DIA_CHI_NAMES[h]} ({h})",
            "menh_cung": DIA_CHI_NAMES[mc], "than_cung": DIA_CHI_NAMES[tc],
            "ban_menh": f"{menh_full} ({menh_el} mệnh)",
            "cuc": cuc_name, "cuc_khac_menh": khac,
            "menh_chu": _menh_chu(mc), "than_chu": _than_chu(chi),
            "am_duong_thuan_nghich": am_duong,
            "tu_tru": {
                "nam":str(lunar_data.year_can_chi),"thang":str(lunar_data.month_can_chi),
                "ngay":str(lunar_data.day_can_chi),"gio":str(lunar_data.hour_can_chi),
            },
        },
        "la_so": la_so,
    }
