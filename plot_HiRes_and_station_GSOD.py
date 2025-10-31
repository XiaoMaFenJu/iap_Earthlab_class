import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import geopandas as gpd

# 文件路径
nc_path = r"D:\data\temp\maqun_HiRes.nc"
csv_path = r"E:\文档\study\地球系统模式\作业及数据\gsod\4157126.csv"

# 读取 HiRes 地表温度数据
ds = xr.open_dataset(nc_path)
ts = ds['TS'] - 273.15  # 开尔文转摄氏度
lat = ds['lat']
lon = ds['lon']

# 读取站点数据，跳过第一行描述信息
df = pd.read_csv(csv_path, skiprows=1)

# 提取并转换温度、经纬度（F -> ℃）
df['lat'] = df.iloc[:, 2]   # C列（索引从0开始）
df['lon'] = df.iloc[:, 3]   # D列
df['temp_C'] = (df.iloc[:, 6] - 32) * 5 / 9  # G列
station_lat = df['lat']
station_lon = df['lon']
station_temp = df['temp_C']

# 设置投影
proj = ccrs.PlateCarree()

# 绘图
fig, ax = plt.subplots(figsize=(10, 8), dpi=300, subplot_kw={'projection': proj})

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

# 绘制等值线填色图
levels = np.arange(np.nanmin(ts), np.nanmax(ts), 1)
contour = ax.contourf(lon, lat, ts, levels=levels, cmap='coolwarm', extend='both', transform=proj)

# 叠加站点温度（边缘黑色，内部颜色与色标一致）
sc = ax.scatter(station_lon, station_lat, c=station_temp, cmap='coolwarm',
                edgecolors='black', linewidths=0.6, s=60, transform=proj)

# 添加色标
cb = plt.colorbar(contour, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
cb.set_label("Surface Temperature (°C)")

# 设置标题
ax.set_title('HiRes 地表温度与站点温度(GSOD)分布图',  font={'family': 'Microsoft YaHei', 'size': 14})

# 显示图像
plt.savefig(r"E:\文档\study\地球系统模式\作业及数据\hires_station_T_GSOD.png")
plt.show()
