import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
from shapely.geometry import MultiPolygon, Polygon


def create_extruded_polygon_map(
    shapefile_path, output_html_path, height_field="total_degr", height_scale=100
):
    """
    创建底面与行政区域一致的3D挤出地图

    参数:
    shapefile_path: Shapefile文件路径
    output_html_path: 输出HTML文件路径
    height_field: 用于决定柱体高度的字段
    height_scale: 高度缩放因子
    """

    try:
        # 1. 读取Shapefile数据
        print("正在读取Shapefile数据...")
        gdf = gpd.read_file(shapefile_path)

        # 检查数据
        if not isinstance(gdf, gpd.GeoDataFrame):
            print("错误: 读取的数据不是GeoDataFrame")
            return None

        print(f"成功读取数据，包含 {len(gdf)} 个要素")
        print(f"数据字段: {list(gdf.columns)}")

        # 2. 检查必要的字段
        if height_field not in gdf.columns:
            print(f"错误: 数据中缺少高度字段 '{height_field}'")
            return None

        # 3. 准备数据 - 将几何图形转换为适合3D挤出的格式
        print("处理几何数据...")

        # 计算高度范围和颜色映射
        height_values = gdf[height_field]
        min_height = height_values.min()
        max_height = height_values.max()

        print(f"高度范围: {min_height} - {max_height}")

        # 准备数据列表
        features = []

        for idx, row in gdf.iterrows():
            geometry = row.geometry
            height = row[height_field]

            # 归一化高度 (0-1)
            normalized_height = (
                (height - min_height) / (max_height - min_height)
                if max_height > min_height
                else 0.5
            )

            # 根据高度设置颜色 (从浅黄到深红)
            if normalized_height < 0.25:
                color = [255, 255, 100, 200]  # 浅黄
            elif normalized_height < 0.5:
                color = [255, 200, 50, 200]  # 橙黄
            elif normalized_height < 0.75:
                color = [255, 150, 0, 200]  # 橙色
            else:
                color = [255, 50, 0, 200]  # 红色

            # 处理不同类型的几何图形
            if geometry.geom_type == "Polygon":
                # 单个多边形
                polygons = [geometry]
            elif geometry.geom_type == "MultiPolygon":
                # 多重多边形 - 取最大的一个
                polygons = list(geometry.geoms)
                # 按面积排序，取最大的
                polygons.sort(key=lambda p: p.area, reverse=True)
                polygons = [polygons[0]] if polygons else []
            else:
                # 跳过非多边形几何
                continue

            for poly in polygons:
                # 获取多边形的外环坐标
                exterior_coords = list(poly.exterior.coords)

                # 确保多边形是闭合的（首尾点相同）
                if exterior_coords[0] != exterior_coords[-1]:
                    exterior_coords.append(exterior_coords[0])

                # 转换为PyDeck需要的格式 [经度, 纬度]
                polygon_coords = [[lon, lat] for lon, lat in exterior_coords]

                features.append(
                    {
                        "polygon": polygon_coords,
                        "height": height * height_scale,
                        "color": color,
                        "name": row.get("NAME", f"区域_{idx}"),
                        "height_value": height,
                        "in_count": row.get("in_count", 0),
                        "out_count": row.get("out_count", 0),
                    }
                )

        if not features:
            print("错误: 没有有效的多边形数据")
            return None

        print(f"处理了 {len(features)} 个多边形")

        # 4. 创建3D挤出地图
        # 计算地图中心点
        all_coords = [coord for feature in features for coord in feature["polygon"]]
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]

        center_lon = np.mean(lons)
        center_lat = np.mean(lats)

        # 创建视图
        view_state = pdk.ViewState(
            longitude=center_lon,
            latitude=center_lat,
            zoom=7,
            pitch=45,  # 倾斜角度，产生3D效果
            bearing=0,
        )

        # 创建多边形挤出层
        polygon_layer = pdk.Layer(
            "PolygonLayer",
            features,
            get_polygon="polygon",
            get_fill_color="color",
            get_elevation="height",
            elevation_scale=1,
            extruded=True,  # 关键参数：启用挤出效果
            pickable=True,
            auto_highlight=True,
            coverage=0.9,
        )

        # 创建文本标注层（可选）
        # 计算每个多边形的中心点用于文本标注
        text_features = []
        for feature in features:
            poly_coords = feature["polygon"]
            if len(poly_coords) > 0:
                # 计算多边形的近似中心
                center_lon = np.mean([coord[0] for coord in poly_coords])
                center_lat = np.mean([coord[1] for coord in poly_coords])

                text_features.append(
                    {"position": [center_lon, center_lat], "text": feature["name"]}
                )

        text_layer = pdk.Layer(
            "TextLayer",
            text_features,
            get_position="position",
            get_text="text",
            get_color=[0, 0, 0, 255],
            get_size=10,
            get_alignment_baseline="bottom",
        )

        # 5. 创建地图
        deck = pdk.Deck(
            layers=[polygon_layer, text_layer],
            initial_view_state=view_state,
            tooltip={
                "html": """
                <b>地区:</b> {name}<br/>
                <b>高度值:</b> {height_value}<br/>
                <b>进入连接:</b> {in_count}<br/>
                <b>外出连接:</b> {out_count}
                """,
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white",
                    "fontSize": "12px",
                },
            },
            map_style="light",
        )

        # 6. 保存为HTML文件
        deck.to_html(output_html_path)
        print(f"3D挤出地图已保存到: {output_html_path}")

        return deck

    except Exception as e:
        print(f"创建地图时出错: {e}")
        import traceback

        traceback.print_exc()
        return None


def create_enhanced_extruded_map(
    shapefile_path, output_html_path, height_field="total_degr"
):
    """
    创建增强版的3D挤出地图，更接近参考图片效果
    """
    try:
        # 读取数据
        gdf = gpd.read_file(shapefile_path)

        # 计算高度和颜色
        height_values = gdf[height_field]
        min_height = height_values.min()
        max_height = height_values.max()

        print(f"增强版 - 高度范围: {min_height} - {max_height}")

        # 准备数据
        features = []

        for idx, row in gdf.iterrows():
            geometry = row.geometry
            height = row[height_field]
            normalized = (
                (height - min_height) / (max_height - min_height)
                if max_height > min_height
                else 0.5
            )

            # 更精细的颜色映射
            if normalized < 0.1:
                color = [255, 255, 200, 220]
            elif normalized < 0.3:
                color = [255, 255, 100, 220]
            elif normalized < 0.5:
                color = [255, 200, 50, 220]
            elif normalized < 0.7:
                color = [255, 150, 0, 220]
            elif normalized < 0.9:
                color = [255, 100, 0, 220]
            else:
                color = [255, 50, 0, 220]

            # 处理几何图形
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

                features.append(
                    {
                        "polygon": polygon_coords,
                        "height": height * 150,  # 调整高度系数
                        "color": color,
                        "name": row.get("NAME", f"区域_{idx}"),
                        "height_value": height,
                        "in_count": row.get("in_count", 0),
                        "out_count": row.get("out_count", 0),
                    }
                )

        # 创建地图
        all_coords = [coord for feature in features for coord in feature["polygon"]]
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]

        view_state = pdk.ViewState(
            longitude=np.mean(lons),
            latitude=np.mean(lats),
            zoom=7.5,
            pitch=50,
            bearing=0,
        )

        polygon_layer = pdk.Layer(
            "PolygonLayer",
            features,
            get_polygon="polygon",
            get_fill_color="color",
            get_elevation="height",
            elevation_scale=1,
            extruded=True,
            pickable=True,
            auto_highlight=True,
            coverage=0.95,
            wireframe=True,  # 显示线框，增强3D效果
            filled=True,
        )

        deck = pdk.Deck(
            layers=[polygon_layer],
            initial_view_state=view_state,
            tooltip={
                "html": """
                <div style="
                    background: rgba(0, 0, 0, 0.8);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 12px;
                ">
                    <b>{name}</b><br/>
                    连接度: {height_value}<br/>
                    进入: {in_count} | 外出: {out_count}
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
    shapefile_path = "~ island.shp"
    output_path = ""
    enhanced_output_path = ""

    # 检查文件是否存在
    import os

    if not os.path.exists(shapefile_path):
        print(f"错误: 文件不存在 - {shapefile_path}")
        print("请检查文件路径是否正确")
    else:
        print(f"找到文件: {shapefile_path}")

        # 创建基础版挤出地图
        print("\n=== 创建基础版3D挤出地图 ===")
        deck1 = create_extruded_polygon_map(shapefile_path, output_path)

        # 创建增强版挤出地图
        print("\n=== 创建增强版3D挤出地图 ===")
        deck2 = create_enhanced_extruded_map(shapefile_path, enhanced_output_path)

        if deck1 is not None:
            print("✓ 基础版挤出地图创建成功")
        else:
            print("✗ 基础版挤出地图创建失败")

        if deck2 is not None:
            print("✓ 增强版挤出地图创建成功")
        else:
            print("✗ 增强版挤出地图创建失败")

        print("\n使用方法:")
        print(f"1. 打开 {output_path} 查看基础版地图")
        print(f"2. 打开 {enhanced_output_path} 查看增强版地图")
        print("3. 在浏览器中可以使用鼠标拖拽旋转、滚轮缩放")
        print("4. 鼠标悬停在柱体上查看详细信息")
