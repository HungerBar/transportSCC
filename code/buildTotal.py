import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
from shapely.geometry import MultiPolygon, Polygon


def process_shapefile_features(
    shapefile_path, height_field="total_degr", height_scale=100, color_scheme="warm"
):
    """
    处理Shapefile数据，返回格式化的特征列表和坐标范围
    """
    try:
        # 读取Shapefile数据
        gdf = gpd.read_file(shapefile_path)

        if not isinstance(gdf, gpd.GeoDataFrame):
            print("错误: 读取的数据不是GeoDataFrame")
            return None, None

        print(f"成功读取数据: {shapefile_path}，包含 {len(gdf)} 个要素")

        # 检查必要的字段
        if height_field not in gdf.columns:
            print(f"错误: 数据中缺少高度字段 '{height_field}'")
            return None, None

        # 计算高度范围
        height_values = gdf[height_field]
        min_height = height_values.min()
        max_height = height_values.max()

        print(f"高度范围: {min_height} - {max_height}")

        features = []
        all_coords = []

        for idx, row in gdf.iterrows():
            geometry = row.geometry
            height = row[height_field]

            # 归一化高度 (0-1)
            normalized_height = (
                (height - min_height) / (max_height - min_height)
                if max_height > min_height
                else 0.5
            )

            # 根据颜色方案设置颜色
            if color_scheme == "warm":
                # 暖色（黄到红）
                if normalized_height < 0.5:
                    color = [255, 200, 50, 200]  # 橙黄
                elif normalized_height < 0.75:
                    color = [255, 150, 0, 200]  # 橙色
                else:
                    color = [255, 50, 0, 200]  # 红色
            elif color_scheme == "cold":
                # 冷色（浅蓝到深蓝）
                if normalized_height < 0.5:
                    color = [0, 80, 255, 200]  # 深蓝
                else:
                    color = [0, 30, 200, 200]  # 藏青
            else:
                # 默认灰色
                color = [150, 150, 150, 200]

            # 处理不同类型的几何图形
            if geometry.geom_type == "Polygon":
                polygons = [geometry]
            elif geometry.geom_type == "MultiPolygon":
                polygons = list(geometry.geoms)
                polygons.sort(key=lambda p: p.area, reverse=True)
                polygons = [polygons[0]] if polygons else []
            else:
                continue

            for poly in polygons:
                exterior_coords = list(poly.exterior.coords)
                if exterior_coords[0] != exterior_coords[-1]:
                    exterior_coords.append(exterior_coords[0])

                polygon_coords = [[lon, lat] for lon, lat in exterior_coords]
                all_coords.extend(polygon_coords)

                features.append(
                    {
                        "polygon": polygon_coords,
                        "height": height * height_scale,
                        "color": color,
                        "name": row.get("NAME", f"区域_{idx}"),
                        "height_value": height,
                        "in_count": row.get("in_count", 0),
                        "out_count": row.get("out_count", 0),
                        "source": shapefile_path.split("/")[-1],
                    }
                )

        if not features:
            print("错误: 没有有效的多边形数据")
            return None, None

        print(f"处理了 {len(features)} 个多边形")
        return features, all_coords

    except Exception as e:
        print(f"处理Shapefile时出错: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def create_extruded_polygon_map(
    shapefile_paths, output_html_path, height_field="total_degr", height_scale=100
):
    """
    创建包含多个Shapefile的3D挤出地图
    shapefile_paths: 包含两个Shapefile路径的列表
    """
    try:
        all_features = []
        all_coords = []
        layers = []

        # 处理第一个Shapefile（暖色）
        features1, coords1 = process_shapefile_features(
            shapefile_paths[0], height_field, height_scale, color_scheme="warm"
        )
        if features1:
            all_features.extend(features1)
            all_coords.extend(coords1)

            # 创建第一个图层（暖色）
            layer1 = pdk.Layer(
                "PolygonLayer",
                features1,
                id="warm_layer",
                get_polygon="polygon",
                get_fill_color="color",
                get_elevation="height",
                elevation_scale=1,
                extruded=True,
                pickable=True,
                auto_highlight=True,
                coverage=0.9,
            )
            layers.append(layer1)

        # 处理第二个Shapefile（冷色）
        if len(shapefile_paths) >= 2:
            features2, coords2 = process_shapefile_features(
                shapefile_paths[1], height_field, height_scale, color_scheme="cold"
            )
            if features2:
                all_features.extend(features2)
                all_coords.extend(coords2)

                # 创建第二个图层（冷色）
                layer2 = pdk.Layer(
                    "PolygonLayer",
                    features2,
                    id="cold_layer",
                    get_polygon="polygon",
                    get_fill_color="color",
                    get_elevation="height",
                    elevation_scale=1,
                    extruded=True,
                    pickable=True,
                    auto_highlight=True,
                    coverage=0.9,
                )
                layers.append(layer2)

        if not all_coords:
            print("错误: 没有有效的坐标数据")
            return None

        # 计算地图中心点
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]
        center_lon = np.mean(lons)
        center_lat = np.mean(lats)

        # 创建视图
        view_state = pdk.ViewState(
            longitude=center_lon,
            latitude=center_lat,
            zoom=7,
            pitch=45,
            bearing=0,
        )

        # 创建文本标注层
        text_features = []
        for feature in all_features:
            poly_coords = feature["polygon"]
            if len(poly_coords) > 0:
                center_lon = np.mean([coord[0] for coord in poly_coords])
                center_lat = np.mean([coord[1] for coord in poly_coords])
                text_features.append(
                    {"position": [center_lon, center_lat], "text": feature["name"]}
                )

        text_layer = pdk.Layer(
            "TextLayer",
            text_features,
            id="text_layer",
            get_position="position",
            get_text="text",
            get_color=[0, 0, 0, 255],
            get_size=10,
            get_alignment_baseline="bottom",
        )
        layers.append(text_layer)

        # 创建地图
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                "html": """
                <div style="background: rgba(0, 0, 0, 0.8); color: white; padding: 10px; border-radius: 5px;">
                    <b>地区:</b> {name}<br/>
                    <b>高度值:</b> {height_value}<br/>
                    <b>进入连接:</b> {in_count}<br/>
                    <b>外出连接:</b> {out_count}<br/>
                    <b>数据源:</b> {source}
                </div>
                """,
                "style": {"fontSize": "12px"},
            },
            map_style="light",
        )

        # 保存为HTML文件
        deck.to_html(output_html_path)
        print(f"\n3D挤出地图已保存到: {output_html_path}")

        return deck

    except Exception as e:
        print(f"创建地图时出错: {e}")
        import traceback

        traceback.print_exc()
        return None


def create_enhanced_extruded_map(
    shapefile_paths, output_html_path, height_field="total_degr"
):
    """
    创建增强版多Shapefile3D挤出地图
    """
    try:
        all_features = []
        all_coords = []
        layers = []

        # 处理第一个Shapefile（暖色）
        features1, coords1 = process_shapefile_features(
            shapefile_paths[0], height_field, height_scale=150, color_scheme="warm"
        )
        if features1:
            all_features.extend(features1)
            all_coords.extend(coords1)

            layer1 = pdk.Layer(
                "PolygonLayer",
                features1,
                id="warm_layer_enhanced",
                get_polygon="polygon",
                get_fill_color="color",
                get_elevation="height",
                elevation_scale=1,
                extruded=True,
                pickable=True,
                auto_highlight=True,
                coverage=0.95,
                wireframe=True,
                filled=True,
            )
            layers.append(layer1)

        # 处理第二个Shapefile（冷色）
        if len(shapefile_paths) >= 2:
            features2, coords2 = process_shapefile_features(
                shapefile_paths[1], height_field, height_scale=150, color_scheme="cold"
            )
            if features2:
                all_features.extend(features2)
                all_coords.extend(coords2)

                layer2 = pdk.Layer(
                    "PolygonLayer",
                    features2,
                    id="cold_layer_enhanced",
                    get_polygon="polygon",
                    get_fill_color="color",
                    get_elevation="height",
                    elevation_scale=1,
                    extruded=True,
                    pickable=True,
                    auto_highlight=True,
                    coverage=0.95,
                    wireframe=True,
                    filled=True,
                )
                layers.append(layer2)

        if not all_coords:
            print("错误: 没有有效的坐标数据")
            return None

        # 计算中心点
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]
        center_lon = np.mean(lons)
        center_lat = np.mean(lats)

        view_state = pdk.ViewState(
            longitude=center_lon,
            latitude=center_lat,
            zoom=7.5,
            pitch=50,
            bearing=0,
        )

        # 创建地图
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                "html": """
                <div style="background: rgba(0, 0, 0, 0.85); color: white; padding: 12px; border-radius: 6px; font-family: Arial;">
                    <b>{name}</b><br/>
                    <span style="color: #ff6b6b;">连接度:</span> {height_value}<br/>
                    <span style="color: #4ecdc4;">进入:</span> {in_count} | <span style="color: #ffe066;">外出:</span> {out_count}<br/>
                    <small>数据源: {source}</small>
                </div>
                """
            },
            map_style="light",
        )

        deck.to_html(output_html_path)
        print(f"增强版3D挤出地图已保存: {output_html_path}")

        return deck

    except Exception as e:
        print(f"创建增强地图时出错: {e}")
        import traceback

        traceback.print_exc()
        return None


# 主程序
if __name__ == "__main__":
    # 请替换为您的实际文件路径
    shapefile_path1 = ""  # 第一个Shapefile（暖色）
    shapefile_path2 = ""  # 第二个Shapefile（冷色）
    output_path = ""
    enhanced_output_path = ""

    shapefile_paths = [shapefile_path1, shapefile_path2]

    # 检查文件是否存在
    valid_paths = []
    for path in shapefile_paths:
        if os.path.exists(path):
            print(f"找到文件: {path}")
            valid_paths.append(path)
        else:
            print(f"警告: 文件不存在 - {path}")

    if len(valid_paths) == 0:
        print("错误: 没有有效的Shapefile文件路径")
    else:
        # 创建基础版组合地图
        print("\n=== 创建基础版组合3D挤出地图 ===")
        deck1 = create_extruded_polygon_map(valid_paths, output_path)

        # 创建增强版组合地图
        print("\n=== 创建增强版组合3D挤出地图 ===")
        deck2 = create_enhanced_extruded_map(valid_paths, enhanced_output_path)

        if deck1 is not None:
            print("✓ 基础版组合地图创建成功")
        else:
            print("✗ 基础版组合地图创建失败")

        if deck2 is not None:
            print("✓ 增强版组合地图创建成功")
        else:
            print("✗ 增强版组合地图创建失败")

        print("\n使用说明:")
        print(f"1. 打开 {output_path} 查看基础版组合地图")
        print(f"2. 打开 {enhanced_output_path} 查看增强版组合地图")
        print("3. 暖色图层（红/黄）来自第一个Shapefile")
        print("4. 冷色图层（蓝）来自第二个Shapefile")
        print("5. 鼠标悬停可查看详细信息，支持拖拽旋转和滚轮缩放")
