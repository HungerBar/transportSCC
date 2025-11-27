import json

import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.geometry as geom
from matplotlib_scalebar.scalebar import ScaleBar


def load_geojson_without_gdal(path):
    """使用 json + shapely 手动解析 GeoJSON（无 fiona/gdal/pyogrio）"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    geometries = []

    for feat in data["features"]:
        props = feat.get("properties", {})
        geometry = geom.shape(feat["geometry"])
        records.append(props)
        geometries.append(geometry)

    return gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")


def add_north_arrow(ax, x, y, size=0.1, text_size=12):
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y + size),
        arrowprops=dict(facecolor="black", width=5, headwidth=15),
        ha="center",
        va="center",
        fontsize=text_size,
        fontweight="bold",
    )


def create_2d_map_with_china_cities(
    data_geojson,
    city_basemap_geojson,
    output_png,
    value_field="total_degr",
    cmap="Blues",
    figsize=(12, 10),
):

    # ===== 加载你的专题数据 =====
    print("加载主体数据（无 GDAL）...")
    gdf = load_geojson_without_gdal(data_geojson)

    # ===== 加载中国城市 GeoJSON 底图 =====
    print("加载中国城市底图（无 GDAL）...")
    cities = load_geojson_without_gdal(city_basemap_geojson)

    # ===== 投影到 Web Mercator =====
    print("投影中...")
    gdf_merc = gdf.to_crs(epsg=3857)
    cities_merc = cities.to_crs(epsg=3857)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # ===== 底图：中国城市 =====
    cities_merc.plot(ax=ax, color="none", edgecolor="gray", linewidth=0.5)

    # ===== 叠加你的专题数据 =====
    gdf_merc.plot(
        ax=ax,
        column=value_field,
        cmap=cmap,
        linewidth=0.8,
        edgecolor="black",
        legend=True,
        legend_kwds={"shrink": 0.6, "label": value_field},
    )

    ax.set_axis_off()

    # ===== 比例尺 =====
    scalebar = ScaleBar(
        1, units="m", dimension="si-length", location="lower left", pad=0.5
    )
    ax.add_artist(scalebar)

    # ===== 指南针 =====
    bounds = gdf.total_bounds
    cx = (bounds[0] + bounds[2]) / 2
    ty = bounds[3]
    add_north_arrow(ax, cx, ty, size=(bounds[3] - bounds[1]) * 0.05)

    # ===== 注记（城市名字） =====
    if "name" in cities.columns:
        for _, row in cities_merc.iterrows():
            pt = row.geometry.representative_point()
            ax.text(pt.x, pt.y, row["name"], fontsize=6, ha="center")

    plt.title("Islands City", fontsize=18)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"地图生成成功：{output_png}")


if __name__ == "__main__":
    data_geojson = ""
    city_basemap_geojson = ""
    output_png = ""

    create_2d_map_with_china_cities(data_geojson, city_basemap_geojson, output_png)
