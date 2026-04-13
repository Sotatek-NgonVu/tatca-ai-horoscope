"""
Pillow-based Tu Vi chart image renderer — Full Edition.

Produces a high-quality 1400×1700 PNG image with:
- 4×4 grid with 12 outer cells (Palaces) and merged center (Thiên Bàn)
- Cung Can prefix on each palace
- Major stars with Miếu/Vượng/Đắc/Bình/Hãm brightness markers
- Color-coded: Blue=Chính tinh, Green=Cát tinh, Red=Sát tinh
- Tràng Sinh stage name + Đại Hạn age in each cell
- Tuần/Triệt markers
- Bác Sĩ ring + full minor star sets
- Center: Tứ Trụ, Mệnh Chủ, Thân Chủ, Nạp Âm, Cục Khắc

Author: Tu Vi Astrology Engine — tatca.ai
"""

from __future__ import annotations

import io
import logging
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.core.interfaces import ChartRendererPort

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
#  Layout Constants
# ═════════════════════════════════════════════════════════════════════════════

IMG_W, IMG_H = 1400, 1700
HEADER_H = 70

GRID_TOP = HEADER_H
GRID_H = IMG_H - HEADER_H
COL_W = IMG_W // 4
ROW_H = GRID_H // 4

# ── Colors ───────────────────────────────────────────────────────────────────
BG_COLOR = (255, 253, 248)
HEADER_BG = (40, 22, 55)
HEADER_TEXT = (255, 215, 0)
GRID_LINE = (90, 70, 110)
CELL_BG = (252, 250, 255)
CENTER_BG = (245, 240, 255)
PALACE_NAME_COLOR = (70, 0, 120)
CHI_COLOR = (110, 80, 140)
CHINH_TINH_COLOR = (20, 45, 140)
CAT_TINH_COLOR = (0, 120, 55)
SAT_TINH_COLOR = (185, 25, 25)
TUAN_TRIET_COLOR = (180, 140, 40)
CENTER_TITLE = (40, 22, 55)
CENTER_TEXT = (55, 45, 75)
CENTER_LABEL = (100, 80, 120)
BORDER_COLOR = (60, 40, 80)
DAI_HAN_COLOR = (140, 110, 160)
TRANG_SINH_COLOR = (160, 130, 50)
THAN_MARKER_COLOR = (200, 50, 50)
CUNG_CAN_COLOR = (100, 80, 130)

# Grid mapping
CHI_TO_GRID: dict[str, tuple[int, int]] = {
    "Tỵ": (0, 0), "Ngọ": (0, 1), "Mùi": (0, 2), "Thân": (0, 3),
    "Thìn": (1, 0),                                "Dậu": (1, 3),
    "Mão": (2, 0),                                  "Tuất": (2, 3),
    "Dần": (3, 0), "Sửu": (3, 1), "Tý": (3, 2),   "Hợi": (3, 3),
}


# ═════════════════════════════════════════════════════════════════════════════
#  Font helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return _load_font(size)


# ═════════════════════════════════════════════════════════════════════════════
#  Renderer
# ═════════════════════════════════════════════════════════════════════════════


class PillowChartRenderer(ChartRendererPort):
    """Renders a full Tu Vi chart as a high-quality PNG."""

    def __init__(self) -> None:
        self._f_header = _load_bold_font(24)
        self._f_palace = _load_bold_font(13)
        self._f_chi = _load_font(11)
        self._f_star_big = _load_bold_font(11)
        self._f_star_sm = _load_font(9)
        self._f_center_title = _load_bold_font(16)
        self._f_center = _load_font(12)
        self._f_center_sm = _load_font(10)
        self._f_meta = _load_font(9)
        self._f_dai_han = _load_font(9)

    def render_chart(self, chart_data: dict[str, Any]) -> bytes:
        info = chart_data.get("thong_tin_chu_nhan", {})
        la_so = chart_data.get("la_so", {})

        img = Image.new("RGB", (IMG_W, IMG_H), BG_COLOR)
        draw = ImageDraw.Draw(img)

        self._draw_header(draw, info)
        self._draw_grid(draw)
        self._draw_center(draw, info)
        self._draw_palaces(draw, la_so)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        logger.info("Chart rendered: %d bytes", buf.getbuffer().nbytes)
        return buf.getvalue()

    def _draw_header(self, draw: ImageDraw.ImageDraw, info: dict) -> None:
        draw.rectangle([0, 0, IMG_W, HEADER_H], fill=HEADER_BG)
        name = info.get("ten", "Lá Số Tử Vi")
        title = f"LÁ SỐ TỬ VI — {name.upper()}"
        bbox = draw.textbbox((0, 0), title, font=self._f_header)
        tw = bbox[2] - bbox[0]
        x = (IMG_W - tw) // 2
        y = (HEADER_H - (bbox[3] - bbox[1])) // 2
        draw.text((x, y), title, fill=HEADER_TEXT, font=self._f_header)

    def _draw_grid(self, draw: ImageDraw.ImageDraw) -> None:
        for r in range(4):
            for c in range(4):
                x0 = c * COL_W
                y0 = GRID_TOP + r * ROW_H
                if r in (1, 2) and c in (1, 2):
                    continue
                draw.rectangle([x0, y0, x0 + COL_W, y0 + ROW_H], fill=CELL_BG)

        # Center
        draw.rectangle(
            [COL_W, GRID_TOP + ROW_H, 3 * COL_W, GRID_TOP + 3 * ROW_H],
            fill=CENTER_BG,
        )
        # Border
        draw.rectangle([0, GRID_TOP, IMG_W - 1, IMG_H - 1], outline=BORDER_COLOR, width=3)
        # Grid lines
        for i in range(1, 4):
            draw.line([(i * COL_W, GRID_TOP), (i * COL_W, IMG_H)], fill=GRID_LINE, width=2)
            draw.line([(0, GRID_TOP + i * ROW_H), (IMG_W, GRID_TOP + i * ROW_H)], fill=GRID_LINE, width=2)

    def _draw_center(self, draw: ImageDraw.ImageDraw, info: dict) -> None:
        cx = COL_W + 15
        cy = GRID_TOP + ROW_H + 12
        mw = 2 * COL_W - 30

        # Title
        t = "LÁ SỐ TỬ VI"
        bbox = draw.textbbox((0, 0), t, font=self._f_center_title)
        draw.text((cx + (mw - (bbox[2] - bbox[0])) // 2, cy), t, fill=CENTER_TITLE, font=self._f_center_title)
        y = cy + 28

        draw.line([(cx + 5, y), (cx + mw - 5, y)], fill=GRID_LINE, width=1)
        y += 8

        # Key info lines
        lines = [
            ("Họ tên", info.get("ten", "—")),
            ("Dương lịch", info.get("ngay_duong_lich", "—")),
            ("Âm lịch", info.get("am_lich", "—")),
            ("Giờ sinh", info.get("gio_sinh", "—")),
            ("", ""),
            ("Âm Dương", info.get("am_duong_thuan_nghich", "—")),
            ("Mệnh", info.get("ban_menh", "—")),
            ("Cục", info.get("cuc", "—")),
        ]

        cuc_khac = info.get("cuc_khac_menh", "")
        if cuc_khac:
            lines.append(("", cuc_khac))

        lines.extend([
            ("", ""),
            ("Mệnh chủ", info.get("menh_chu", "—")),
            ("Thân chủ", info.get("than_chu", "—")),
        ])

        for label, value in lines:
            if not label and not value:
                y += 5
                continue
            if label:
                draw.text((cx + 8, y), f"{label}:", fill=CENTER_LABEL, font=self._f_center_sm)
                draw.text((cx + 100, y), value, fill=CENTER_TEXT, font=self._f_center_sm)
            else:
                draw.text((cx + 20, y), value, fill=SAT_TINH_COLOR, font=self._f_center_sm)
            y += 16

        # Tứ Trụ
        tu_tru = info.get("tu_tru")
        if tu_tru:
            y += 8
            draw.line([(cx + 5, y), (cx + mw - 5, y)], fill=GRID_LINE, width=1)
            y += 6
            draw.text((cx + 8, y), "TỨ TRỤ:", fill=CENTER_TITLE, font=self._f_center_sm)
            y += 18

            pillars = [
                ("Năm", tu_tru.get("nam", "—")),
                ("Tháng", tu_tru.get("thang", "—")),
                ("Ngày", tu_tru.get("ngay", "—")),
                ("Giờ", tu_tru.get("gio", "—")),
            ]
            col_w = mw // 4
            for i, (label, value) in enumerate(pillars):
                px = cx + 8 + i * col_w
                draw.text((px, y), label, fill=CENTER_LABEL, font=self._f_center_sm)
                draw.text((px, y + 13), value, fill=CENTER_TEXT, font=self._f_center_sm)

    def _draw_palaces(self, draw: ImageDraw.ImageDraw, la_so: dict) -> None:
        for palace_name, data in la_so.items():
            chi = data.get("dia_chi", "")
            pos = CHI_TO_GRID.get(chi)
            if pos is None:
                continue
            row, col = pos
            cx = col * COL_W
            cy = GRID_TOP + row * ROW_H
            self._draw_palace(draw, cx, cy, COL_W, ROW_H, palace_name, data)

    def _draw_palace(
        self, draw: ImageDraw.ImageDraw,
        x: int, y: int, w: int, h: int,
        name: str, data: dict,
    ) -> None:
        pad = 5
        chi = data.get("dia_chi", "")
        cung_can = data.get("cung_can", "")
        chinh = data.get("chinh_tinh", [])
        cat = data.get("cat_tinh", [])
        sat = data.get("sat_tinh", [])
        tuan_triet = data.get("tuan_triet", "")
        dai_han = data.get("dai_han", "")
        trang_sinh = data.get("trang_sinh", "")
        is_than = data.get("is_than", False)

        # ── Top-left: Cung Can + Chi ─────────────────────────────────
        can_chi_text = f"{cung_can[0]}.{chi}" if cung_can else chi
        draw.text((x + pad, y + pad), can_chi_text, fill=CUNG_CAN_COLOR, font=self._f_chi)

        # ── Top-center: Palace name ──────────────────────────────────
        pname = name
        if is_than:
            pname = f"{name} (Thân)"
        bbox = draw.textbbox((0, 0), pname, font=self._f_palace)
        pw = bbox[2] - bbox[0]
        draw.text((x + (w - pw) // 2, y + pad), pname, fill=PALACE_NAME_COLOR, font=self._f_palace)

        # ── Top-right: Đại Hạn ───────────────────────────────────────
        if dai_han:
            dh_bbox = draw.textbbox((0, 0), dai_han, font=self._f_dai_han)
            dw = dh_bbox[2] - dh_bbox[0]
            draw.text(
                (x + w - dw - pad, y + pad + 2),
                dai_han, fill=DAI_HAN_COLOR, font=self._f_dai_han,
            )

        # ── Major stars (bold, blue, centered) ───────────────────────
        sy = y + pad + 20
        line_h_big = 14
        line_h_sm = 12

        for star in chinh[:3]:
            if sy + line_h_big > y + h - 20:
                break
            sbbox = draw.textbbox((0, 0), star, font=self._f_star_big)
            sw = sbbox[2] - sbbox[0]
            draw.text((x + (w - sw) // 2, sy), star, fill=CHINH_TINH_COLOR, font=self._f_star_big)
            sy += line_h_big

        # Separator after major stars
        if chinh and (cat or sat):
            sy += 3

        # ── Minor stars: left=Cát (green), right=Sát (red) ──────────
        max_minor = 8
        cat_y = sy
        for star in cat[:max_minor]:
            if cat_y + line_h_sm > y + h - 18:
                break
            draw.text((x + pad, cat_y), star, fill=CAT_TINH_COLOR, font=self._f_star_sm)
            cat_y += line_h_sm

        sat_y = sy
        for star in sat[:max_minor]:
            if sat_y + line_h_sm > y + h - 18:
                break
            sbbox = draw.textbbox((0, 0), star, font=self._f_star_sm)
            sw = sbbox[2] - sbbox[0]
            draw.text((x + w - sw - pad, sat_y), star, fill=SAT_TINH_COLOR, font=self._f_star_sm)
            sat_y += line_h_sm

        # ── Bottom-left: Tràng Sinh ──────────────────────────────────
        if trang_sinh:
            draw.text(
                (x + pad, y + h - 15),
                trang_sinh, fill=TRANG_SINH_COLOR, font=self._f_meta,
            )

        # ── Bottom-right: Tuần / Triệt ──────────────────────────────
        if tuan_triet:
            tt_bbox = draw.textbbox((0, 0), tuan_triet, font=self._f_meta)
            tw = tt_bbox[2] - tt_bbox[0]
            draw.text(
                (x + w - tw - pad, y + h - 15),
                tuan_triet, fill=TUAN_TRIET_COLOR, font=self._f_meta,
            )
