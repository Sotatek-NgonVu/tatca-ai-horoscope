import os
import json
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader

class HtmlChartRenderer:
    """Renders a Tu Vi chart layout as a beautiful HTML file."""

    def __init__(self, template_dir: str = None):
        if not template_dir:
            template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
        
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template_name = "tuvi_template.html"

    def render_chart(self, chart_data: Dict[str, Any], output_path: str = "laso.html") -> str:
        """
        Takes the chart JSON dictionary and writes out an HTML file.
        Returns the absolute path to the generated HTML file.
        """
        template = self.env.get_template(self.template_name)
        
        # Prepare data for template
        template_kwargs = {
            "ten": chart_data["thong_tin_chu_nhan"].get("ten", "Không rõ"),
            "ngay_duong_lich": chart_data["thong_tin_chu_nhan"].get("ngay_duong_lich"),
            "am_lich": chart_data["thong_tin_chu_nhan"].get("am_lich"),
            "tu_tru": chart_data["thong_tin_chu_nhan"].get("tu_tru", {}),
            "am_duong_nam_sinh": chart_data["thong_tin_chu_nhan"].get("am_duong_nam_sinh", ""),
            "am_duong_menh": chart_data["thong_tin_chu_nhan"].get("am_duong_menh", ""),
            "ban_menh": chart_data["thong_tin_chu_nhan"].get("ban_menh", ""),
            "cuc": chart_data["thong_tin_chu_nhan"].get("cuc", ""),
            "sinh_khac_cuc_menh": chart_data["thong_tin_chu_nhan"].get("sinh_khac_cuc_menh", ""),
            "menh_chu": chart_data["thong_tin_chu_nhan"].get("menh_chu", ""),
            "than_chu": chart_data["thong_tin_chu_nhan"].get("than_chu", ""),
            "laso": {}
        }

        # Map la_so dictionary by Dia Chi (Tý, Sửu, Dần...)
        for ten_cung, cung in chart_data["la_so"].items():
            dia_chi = cung.get("dia_chi")
            if dia_chi:
                cung["ten_cung"] = ten_cung # Inject ten_cung into the dict for the template
                template_kwargs["laso"][dia_chi] = cung

        html_content = template.render(**template_kwargs)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return os.path.abspath(output_path)
