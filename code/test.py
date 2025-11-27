import geopandas as gpd

# 读取 Shapefile 文件
gdf = gpd.read_file("scc/SCC.shp")  # 请将路径替换为你的 .shp 文件实际路径

# 查看所有字段（词条）信息
print("字段名称和类型:")
print(gdf.dtypes)

print("\n前几行数据（包含所有属性）:")
print(gdf.head())

# 单独查看字段名称列表
print("\n字段名称列表:")
for col in gdf.columns:
    print(f"- {col}")
