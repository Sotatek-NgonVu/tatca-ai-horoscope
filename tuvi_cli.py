import sys
import os

# Đảm bảo có thể import được package `app`
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.domain.models import BirthData, Gender
from app.services.tuvi_engine import LasoTuViEngine
from app.services.chart_renderer import PillowChartRenderer
import json

def get_gender(user_input: str) -> Gender:
    user_input = user_input.strip().lower()
    if user_input in ["nu", "nữ", "female", "f"]:
        return Gender.FEMALE
    return Gender.MALE

def main():
    print("=" * 50)
    print("      CÔNG CỤ LẬP LÁ SỐ TỬ VI (CLI)")
    print("=" * 50)
    
    # 1. Thu thập thông tin
    name = input("▶ Nhập họ tên [Mặc định: Nguyễn Văn A]: ").strip()
    if not name:
        name = "Nguyễn Văn A"
        
    dob_str = input("▶ Nhập ngày sinh Dương lịch (YYYY-MM-DD) [VD: 1990-05-15]: ").strip()
    if not dob_str:
        dob_str = "1990-05-15"
        
    gender_input = input("▶ Nhập giới tính (nam / nu) [Mặc định: nam]: ").strip()
    gender = get_gender(gender_input)
    
    print("\n[BẢNG GIỜ SINH ÂM LỊCH]")
    print("  0: Tý (23h-01h)     4: Thìn (07h-09h)     8: Thân (15h-17h)")
    print("  1: Sửu (01h-03h)    5: Tỵ   (09h-11h)     9: Dậu  (17h-19h)")
    print("  2: Dần (03h-05h)    6: Ngọ  (11h-13h)    10: Tuất (19h-21h)")
    print("  3: Mão (05h-07h)    7: Mùi  (13h-15h)    11: Hợi  (21h-23h)")
    hour_input = input("▶ Nhập số tương ứng với giờ sinh (0-11) [Mặc định: 3 (Mão)]: ").strip()
    try:
        birth_hour = int(hour_input)
        if not (0 <= birth_hour <= 11):
            birth_hour = 3
    except ValueError:
        birth_hour = 3

    print("\nĐang tính toán lá số...")
    
    # 2. Khởi tạo đối tượng BirthData và Engine
    birth_data = BirthData(
        name=name,
        solar_dob=dob_str,
        gender=gender,
        birth_hour=birth_hour
    )
    
    engine = LasoTuViEngine()
    
    try:
        # Lấy dữ liệu dạng JSON dict
        chart = engine.generate_chart(birth_data)
        
        print("\n" + "=" * 50)
        print("KẾT QUẢ TÍNH TOÁN")
        print("=" * 50)
        chu_nhan = chart.get("thong_tin_chu_nhan", {})
        for k, v in chu_nhan.items():
            if isinstance(v, dict):
                print(f"- {k}:")
                for sub_k, sub_v in v.items():
                    print(f"    {sub_k}: {sub_v}")
            else:
                print(f"- {k}: {v}")
                
        # 3. Vẽ hình ảnh lá số
        renderer = PillowChartRenderer()
        img_bytes = renderer.render_chart(chart)
        
        file_name = f"laso_{name.replace(' ', '_')}.png"
        out_path = os.path.join(os.getcwd(), file_name)
        
        with open(out_path, "wb") as f:
            f.write(img_bytes)
            
        print(f"\n✅ Đã tạo thành công! Hình ảnh lá số được lưu tại:")
        print(f"👉 {out_path}")
        
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình tạo lá số: {e}")

if __name__ == "__main__":
    main()
