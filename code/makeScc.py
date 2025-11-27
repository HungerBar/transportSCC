import os
from pathlib import Path

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point


def load_data(city_shp_path, route_shp_path):
    """
    读取城市点数据和铁路航线线数据
    """
    gdf_cities = gpd.read_file(city_shp_path)
    gdf_routes = gpd.read_file(route_shp_path)
    return gdf_cities, gdf_routes


def build_city_network(gdf_cities, gdf_routes):
    """
    基于铁路航线数据构建城市连通图
    """
    G = nx.DiGraph()

    # 添加城市节点
    for idx, city in gdf_cities.iterrows():
        city_id = city.get("city_id", idx)
        G.add_node(city_id, geometry=city.geometry, properties=city.to_dict())

    # 坐标转换和缓冲区处理
    if gdf_cities.crs and gdf_cities.crs.is_geographic:
        gdf_cities_projected = gdf_cities.to_crs("EPSG:4527")  # 使用适合中国区域的投影
    else:
        gdf_cities_projected = gdf_cities.copy()

    # 建立空间缓冲区用于连接匹配（单位：米）
    buffer_distance = 10000  # 10公里缓冲区
    gdf_cities_buffered = gdf_cities_projected.copy()
    gdf_cities_buffered["geometry"] = gdf_cities_projected.geometry.buffer(
        buffer_distance
    )

    # 查找每条线路连接的起点和终点城市
    for route_idx, route in gdf_routes.iterrows():
        route_line = route.geometry
        if not isinstance(route_line, LineString):
            continue

        # 获取线路端点
        start_point = Point(route_line.coords[0])
        end_point = Point(route_line.coords[-1])

        # 将端点转换到与城市相同的坐标系
        start_point_projected = (
            gpd.GeoSeries([start_point], crs=gdf_routes.crs)
            .to_crs(gdf_cities_projected.crs)
            .iloc[0]
        )
        end_point_projected = (
            gpd.GeoSeries([end_point], crs=gdf_routes.crs)
            .to_crs(gdf_cities_projected.crs)
            .iloc[0]
        )

        # 空间查询匹配起点城市
        start_cities = gdf_cities_buffered[
            gdf_cities_buffered.intersects(start_point_projected)
        ]
        end_cities = gdf_cities_buffered[
            gdf_cities_buffered.intersects(end_point_projected)
        ]

        if not start_cities.empty and not end_cities.empty:
            start_city_id = start_cities.iloc[0].get("city_id", start_cities.index[0])
            end_city_id = end_cities.iloc[0].get("city_id", end_cities.index[0])

            # 避免自环
            if start_city_id != end_city_id:
                G.add_edge(start_city_id, end_city_id)

    return G


def compute_scc(G):
    """
    计算图的强连通组件
    返回SCC编号映射字典和组件列表
    """
    scc_list = list(nx.strongly_connected_components(G))
    scc_mapping = {}
    for scc_id, component in enumerate(scc_list):
        for city_id in component:
            scc_mapping[city_id] = scc_id
    return scc_mapping, scc_list


def export_scc_cities(gdf_cities, scc_mapping, output_path):
    """
    只输出属于SCC的城市到Shapefile
    """
    # 创建结果副本
    result_gdf = gdf_cities.copy()

    # 添加SCC标识
    result_gdf["scc_id"] = result_gdf.get("city_id", result_gdf.index).map(scc_mapping)

    # 关键修改：只保留属于SCC的城市（scc_id不为NaN）
    scc_cities_gdf = result_gdf[result_gdf["scc_id"].notna()].copy()

    if len(scc_cities_gdf) == 0:
        print("警告：没有找到属于任何强连通组件的城市")
        return scc_cities_gdf

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 输出到Shapefile
    scc_cities_gdf.to_file(output_path, driver="ESRI Shapefile", encoding="utf-8")

    print(f"成功输出 {len(scc_cities_gdf)} 个属于SCC的城市到: {output_path}")
    print(f"这些城市分布在 {scc_cities_gdf['scc_id'].nunique()} 个不同的强连通组件中")

    return scc_cities_gdf


def main(city_shp_path, route_shp_path, output_shp_path):
    """
    主函数：只处理并输出属于SCC的城市
    """
    # 读取数据
    gdf_cities, gdf_routes = load_data(city_shp_path, route_shp_path)
    print(f"加载城市数量: {len(gdf_cities)}")
    print(f"加载航线数量: {len(gdf_routes)}")

    # 构建网络
    G = build_city_network(gdf_cities, gdf_routes)
    print(f"网络包含 {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

    # 计算SCC
    scc_mapping, scc_components = compute_scc(G)
    print(f"发现 {len(scc_components)} 个强连通组件")

    # 只输出属于SCC的城市
    scc_cities = export_scc_cities(gdf_cities, scc_mapping, output_shp_path)

    # 输出统计信息
    if len(scc_cities) > 0:
        component_sizes = scc_cities.groupby("scc_id").size()
        print("\n各SCC组件大小分布:")
        for scc_id, size in component_sizes.items():
            print(f"组件 {scc_id}: 包含 {size} 个城市")

    return scc_cities, scc_components


if __name__ == "__main__":
    # 使用绝对路径避免文件路径问题
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    # 文件路径配置
    city_shapefile = (
        "/Users/qjxie/Github/Gis/project1/data/City/CN_city.shp"  # 城市点数据
    )
    route_shapefile = "/Users/qjxie/Github/Gis/project1/data/transport/china_transport_network.shp"  # 铁路航线线数据
    output_shapefile = "code/china_cities_scc.shp"  # 输出结果

    # 确保输出目录存在
    (project_root / "output").mkdir(exist_ok=True)

    # 执行分析
    scc_cities, components = main(city_shapefile, route_shapefile, output_shapefile)
