import os

import geopandas as gpd
from shapely.geometry import Point


def classify_cities_by_connections(
    transport_shp, cities_shp, scc_output, island_output, k=10000, m=5
):
    """
    根据进出线路数对城市分类：
    SCC: >= k 条线路进出
    Island: < m 条线路进出
    """
    print("读取城市数据...")
    cities = gpd.read_file(cities_shp)
    cities = cities[cities.geometry.notnull()].copy()
    cities = cities.to_crs("EPSG:4326")

    print("读取交通网络数据...")
    transport = gpd.read_file(transport_shp)
    transport = transport[transport.geometry.notnull()].copy()
    transport = transport.to_crs("EPSG:4326")

    # 初始化进出计数
    cities["in_count"] = 0
    cities["out_count"] = 0
    cities["total_degree"] = 0

    print("统计每个城市的进出线路数...")
    for idx, city in cities.iterrows():
        city_point = city.geometry
        # 统计起点线路
        out_count = transport.geometry.apply(
            lambda line: line is not None
            and line.geom_type == "LineString"
            and Point(line.coords[0]).distance(city_point) < 1e-6
        ).sum()
        # 统计终点线路
        in_count = transport.geometry.apply(
            lambda line: line is not None
            and line.geom_type == "LineString"
            and Point(line.coords[-1]).distance(city_point) < 1e-6
        ).sum()
        cities.at[idx, "in_count"] = in_count
        cities.at[idx, "out_count"] = out_count
        cities.at[idx, "total_degree"] = in_count + out_count

    print("根据阈值分类城市...")
    scc_cities = cities[cities["total_degree"] >= k].copy()
    island_cities = cities[cities["total_degree"] < m].copy()

    # 保存结果
    os.makedirs(os.path.dirname(scc_output), exist_ok=True)
    os.makedirs(os.path.dirname(island_output), exist_ok=True)

    scc_cities.to_file(scc_output, driver="ESRI Shapefile", encoding="utf-8")
    island_cities.to_file(island_output, driver="ESRI Shapefile", encoding="utf-8")

    print(f"SCC城市保存到: {scc_output}，共 {len(scc_cities)} 个")
    print(f"island城市保存到: {island_output}，共 {len(island_cities)} 个")

    return scc_cities, island_cities


if __name__ == "__main__":
    transport_shp = "path To Transport"
    cities_shp = "path to cities"
    scc_output = "path to sccOut"
    island_output = "path to islandOut"

    # 设置阈值
    k = 42  # SCC 最少进出线路数
    m = 6  # island 最大进出线路数

    scc, island = classify_cities_by_connections(
        transport_shp, cities_shp, scc_output, island_output, k, m
    )
