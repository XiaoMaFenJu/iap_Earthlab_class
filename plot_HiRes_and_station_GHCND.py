import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import geopandas as gpd

# -----------------------
# 1. 读取 HiRes 网格数据
# -----------------------
nc_path = r"D:\data\temp\maqun_HiRes.nc"
ds = xr.open_dataset(nc_path)

lon = ds['lon'].values
lat = ds['lat'].values
ts = ds['TS'].values - 273.15  # 转换为 °C

# -----------------------
# 2. 读取站点数据 (使用固定宽度)
# -----------------------
txt_path = r"D:\data\temp\4156394.txt"

# 根据文件的列间距定义宽度（可根据实际略调）
colspecs = [
    (0, 17),     # STATION
    (18, 68),    # STATION_NAME
    (69, 79),    # ELEVATION
    (80, 90),    # LATITUDE
    (91, 101),   # LONGITUDE
    (102, 110),  # DATE
    (111, 119),  # TAVG
]

names = ["STATION", "STATION_NAME", "ELEVATION", "LATITUDE", "LONGITUDE",
         "DATE", "TAVG"]

df = pd.read_fwf(txt_path, colspecs=colspecs, skiprows=2, names=names)

# 清洗温度
df = df.dropna(subset=['TAVG'])
df['TAVG'] = pd.to_numeric(df['TAVG'], errors='coerce')
df = df.dropna(subset=['TAVG'])

# 转换温度单位
df['TAVG_C'] = (df['TAVG'] - 32) * 5/9  # °F → °C

# -----------------------
# 3. 绘图
# -----------------------
fig = plt.figure(figsize=(10, 8),dpi=300)
proj = ccrs.PlateCarree()
ax = plt.axes(projection=proj)

# 设置范围
extent = [lon.min(), lon.max(), lat.min(), lat.max()]
ax.set_extent(extent, crs=proj)

# 添加地理要素
ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor='none')
# ax.add_feature(cfeature.RIVERS, edgecolor='lightblue')
china_boundary = gpd.read_file(
    r"D:\data\shp\China_shp_2504\2025年4月中国国界、省界、市界地图shp\国界.shp"
)
# 绘制中国国界（新增部分）
ax.add_geometries(
    china_boundary.geometry,  # 提取几何信息
    crs=proj,  # 与地图投影一致
    edgecolor='black',  # 国界颜色
    facecolor='none',  # 不填充
    linewidth=0.7  # 线宽
)

# -----------------------
# 4. 绘制 HiRes 等值线填色
# -----------------------
levels = np.arange(np.nanmin(ts), np.nanmax(ts), 1)
cmap = plt.get_cmap('coolwarm')

cf = ax.contourf(lon, lat, ts, levels=levels, cmap=cmap, extend='both', transform=proj)
# 添加色标
cb = plt.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
cb.set_label("Surface Temperature (°C)")

# -----------------------
# 5. 绘制站点数据
# -----------------------
sc = ax.scatter(df['LONGITUDE'], df['LATITUDE'], c=df['TAVG_C'],
                cmap=cmap, edgecolors='black', linewidths=0.8,
                s=50, transform=proj)

# -----------------------
# 6. 标题与显示
# -----------------------
# 设置标题
ax.set_title('HiRes 地表温度与站点温度(GHCND)分布图',  font={'family': 'Microsoft YaHei', 'size': 14})

# 显示图像
plt.savefig(r"E:\文档\study\地球系统模式\作业及数据\hires_station_T_GHCND.png")
plt.show()
