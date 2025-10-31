import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import geopandas as gpd

# ------------------------------
# 1. 读取 HiRes 网格数据
# ------------------------------
ncfile = r"D:\data\temp\maqun_HiRes.nc"
print(os.path.exists(ncfile))
# with open(ncfile, 'rb') as f:
ds = xr.open_dataset(ncfile)

# 假设变量名为 PS（地表温度），经纬度为 lat、lon
temp = ds["TS"].values - 273.15
lat = ds["lat"]
lon = ds["lon"]
print(np.max(lat), np.min(lat))
print(np.max(lon), np.min(lon))
# ------------------------------
# 2. 读取并筛选站点数据
# ------------------------------
txtfile = r"D:\data\temp\SURF_CLI_CHN_MUL_DAY-TEM-12001-200809.TXT"

# 读取为定宽文本（以空格分隔，注意数据间可能有不规则空格）
colnames = ["StationID", "Lat", "Lon", "Alt", "Year", "Month", "Day",
            "Tmean", "Tmax", "Tmin", "QCmean", "QCmax", "QCmin"]


df = pd.read_csv(txtfile, delim_whitespace=True, names=colnames)

# 筛选 2008年9月9日 的数据
df = df[(df["Year"] == 2008) & (df["Month"] == 9) & (df["Day"] == 9)]

# 经纬度从度分转换为十进制度
# 例如 5258 -> 52°58' = 52 + 58/60
def dm2dd(x):
    deg = np.floor(x / 100)
    minute = x % 100
    return deg + minute / 60

df["Lat_dd"] = df["Lat"].apply(dm2dd)
df["Lon_dd"] = df["Lon"].apply(dm2dd)

# 转换温度单位为 °C（原数据单位为0.1°C）
df["Tmean_C"] = df["Tmean"] / 10.0

# ------------------------------
# 3. 绘制等值线填色图 + 站点数据
# ------------------------------
# 设置中文显示
# plt.rcParams["font.family"] = ["Microsoft YaHei"]
# plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

fig = plt.figure(figsize=(10, 8),dpi=300)
proj = ccrs.PlateCarree()
ax = plt.axes(projection=proj)

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

# 绘制网格数据的等值线填色图
levels = np.linspace(temp.min(), temp.max(), 20)
c = ax.contourf(lon, lat, temp, levels=levels, cmap='coolwarm', extend='both', transform=ccrs.PlateCarree())

# 叠加站点观测数据（圆点，边缘黑色，填色对应温度）
sc = plt.scatter(df["Lon_dd"], df["Lat_dd"], c=df["Tmean_C"], cmap='coolwarm',
                 edgecolors='black', s=20, transform=ccrs.PlateCarree(), vmin=temp.min(), vmax=temp.max())

# 添加色标
cb = plt.colorbar(c, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
cb.set_label("Surface Temperature (°C)")

# 设置标题
ax.set_title("2008-09-09 HiRes 地表温度 + 站点温度分布",  font={'family': 'Microsoft YaHei', 'size': 14})

# 可选：设置显示区域（如中国区域）
# ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())

plt.savefig(r"E:\文档\study\地球系统模式\作业及数据\hires_station_T.png")
plt.show()