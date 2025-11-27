from pathlib import Path

from ababba import (
    process_china_transport_data,
)  # ← 将之前我给你的函数保存为 your_file.py

world_routes_path = ""  # 航线
china_railways_path = ""  # 铁路
china_boundary_path = ""  # 国界

output_dir = ""
Path(output_dir).mkdir(parents=True, exist_ok=True)

output_path = f"{output_dir}/china_transport_network.shp"

combined = process_china_transport_data(
    world_routes_file=world_routes_path,
    china_railways_file=china_railways_path,
    china_boundary_file=china_boundary_path,  # 推荐使用国界判断相交
    output_file=output_path,
)

print(combined.head())
print("处理完成！")
