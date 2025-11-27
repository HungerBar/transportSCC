from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union


def make_unique_columns(df):
    """保证字段名唯一"""
    cols = df.columns.tolist()
    new_cols = []
    used = set()
    for c in cols:
        new_c = c
        i = 1
        while new_c in used:
            new_c = f"{c}_{i}"
            i += 1
        new_cols.append(new_c)
        used.add(new_c)
    df.columns = new_cols
    return df


def align_columns(gdf_list):
    """对齐所有 GeoDataFrame 的字段（字段并集），缺的补 None"""
    # 字段并集
    all_cols = set()
    for g in gdf_list:
        all_cols |= set(g.columns)

    # 每个 gdf 补缺字段
    gdf_new = []
    for g in gdf_list:
        g2 = g.copy()
        for c in all_cols:
            if c not in g2.columns:
                g2[c] = None
        gdf_new.append(g2[sorted(all_cols)])
    return gdf_new


def shorten_columns_to_shapefile(df):
    """Shapefile 字段名≤10字符，自动截断并保持唯一"""
    mapping = {}
    new_cols = []
    used = set()

    for c in df.columns:
        short = c[:10]  # 截断
        base = short
        i = 1
        while short in used:
            short = f"{base[:8]}{i}"
            i += 1
        mapping[c] = short
        used.add(short)
        new_cols.append(short)

    df.columns = new_cols
    return df


def process_china_transport_data(
    world_routes_file,
    china_railways_file,
    output_file,
    china_boundary_file=None,
):
    """
    修复后的航线+铁路处理函数，支持：
    - Great Circle 航线相交
    - 字段对齐（不丢字段）
    - 自动字段名唯一
    - Shapefile 字段名 ≤ 10 字符
    """

    print("读取数据...")
    world = gpd.read_file(world_routes_file)
    rail = gpd.read_file(china_railways_file)

    # 读取中国真实边界（推荐 natural earth 或国界 shp）
    if china_boundary_file:
        china_border = gpd.read_file(china_boundary_file).to_crs("EPSG:4326")
        china_geom = unary_union(china_border.geometry)
    else:
        print("⚠ 未提供中国真实边界，使用 bbox 替代，不建议用于大圆航线")
        from shapely.geometry import box

        china_geom = box(73.66, 18.16, 135.05, 53.56)

    # CRS
    world = world.to_crs("EPSG:4326")
    rail = rail.to_crs("EPSG:4326")

    print("筛选航线：使用 intersects() 而非 bbox")
    # Great Circle 航线可能跨经度 → 不能用 bounds
    world_china = world[world.geometry.intersects(china_geom)].copy()

    print(f"保留航线: {len(world_china)} 条")

    print("筛选铁路...")
    rail_china = rail[rail.geometry.intersects(china_geom)].copy()
    print(f"保留铁路: {len(rail_china)} 条")

    # 标记类型
    world_china["data_type"] = "air_route"
    rail_china["data_type"] = "railway"

    # 确保字段名唯一
    world_china = make_unique_columns(world_china)
    rail_china = make_unique_columns(rail_china)

    # 字段对齐：字段并集，不丢字段
    world_aligned, rail_aligned = align_columns([world_china, rail_china])

    # 合并
    combined = gpd.GeoDataFrame(
        pd.concat([world_aligned, rail_aligned], ignore_index=True),
        crs="EPSG:4326",
    )

    print("裁剪字段名以兼容 Shapefile（≤10 字符）")
    combined = shorten_columns_to_shapefile(combined)

    print(f"保存到 {output_file}")
    combined.to_file(output_file, driver="ESRI Shapefile", encoding="utf-8")

    print("完成！")
    return combined
