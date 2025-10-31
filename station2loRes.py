import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

# ============ 1. 读取站点 TXT 数据 ============

txt_path = r"E:\文档\study\地球系统模式\作业及数据\SURF_CLI_CHN_MUL_DAY-TEM-12001-200809.TXT"

# 自定义读取：固定列宽或分隔符（这里假设空格分隔）
col_names = [
    "station_id", "lat_raw", "lon_raw", "alt", "year", "month", "day",
    "tmean", "tmax", "tmin", "qc_tmean", "qc_tmax", "qc_tmin"
]

# df = pd.read_csv(txt_path, delim_whitespace=True, header=None, names=col_names)
with open(txt_path, 'rb') as f:
    df = pd.read_csv(txt_path, delim_whitespace=True, header=None, names=col_names)
    # print(df.dims)

# 转换经纬度：如5258 → 52 + 58/60 = 52.9667
def degmin_to_deg(val):
    deg = val // 100
    minute = val % 100
    return deg + minute / 60.0

df["lat"] = df["lat_raw"].apply(degmin_to_deg)
df["lon"] = df["lon_raw"].apply(degmin_to_deg)
df["tmean_C"] = df["tmean"] / 10.0  # 转为摄氏度

# ============ 2. 选取2008年9月9日的数据 ============
df_911 = df[(df["year"] == 2008) & (df["month"] == 9) & (df["day"] == 9)].copy()

if df_911.empty:
    raise ValueError("未找到2008年9月9日的站点数据，请检查TXT文件。")

# ============ 3. 读取低分辨率网格 ============
dem_path = r"E:\文档\study\地球系统模式\作业及数据\DemLoRes.nc"
# ds_dem = xr.open_dataset(dem_path) # 有中文路径不能直接打开

with open(dem_path, 'rb') as f:
    ds_dem = xr.open_dataset(f)
    print(ds_dem.dims)

lats = ds_dem["lat"].values
lons = ds_dem["lon"].values
lon2d, lat2d = np.meshgrid(lons, lats)

print(np.min(df["lat"]),np.max(df["lat"]))
print(np.min(df["lon"]),np.max(df["lon"]))
print(np.min(lats),np.max(lats))
print(np.min(lons),np.max(lons))
'''
站点数据
16.533333333333335 53.46666666666667
75.23333333333333 134.28333333333333
低分辨率
20.249999999997 35.749999999997
88.25 111.75
高分辨率
10 55
70 140

'''

# ============ 4. 空间插值 ============
points = np.column_stack((df_911["lon"], df_911["lat"]))
values = df_911["tmean_C"].values

# 线性插值（适合气象数据）
grid_tmean = griddata(points, values, (lon2d, lat2d), method='linear')

# 对于插值外的NaN，用最近邻填补
mask_nan = np.isnan(grid_tmean)
if np.any(mask_nan):
    grid_tmean[mask_nan] = griddata(points, values, (lon2d[mask_nan], lat2d[mask_nan]), method='nearest')

# ============ 5. 保存为 NetCDF 文件 ============
out_path = r"E:\文档\study\地球系统模式\作业及数据\maqun_LoRes.nc"

ds_out = xr.Dataset(
    {
        "SAT": (("lat", "lon"), grid_tmean)
    },
    coords={
        "lat": lats,
        "lon": lons
    }
)

ds_out["SAT"].attrs["units"] = "°C"
ds_out["SAT"].attrs["long_name"] = "Daily mean surface air temperature (interpolated)"
ds_out.attrs["description"] = "Interpolated 2008-09-11 surface air temperature to low-res DEM grid"
ds_out.attrs["source_data"] = txt_path
ds_out.attrs["method"] = "scipy.griddata (linear + nearest fill)"


with open(out_path.encode('utf-8'), "wb") as f:
    ds_out.to_netcdf(f)

print("✅ 插值完成，结果已保存为：", out_path)
