import xarray as xr
import numpy as np
from scipy.interpolate import griddata

# ==== 文件路径 ====
cesm_path = r"/data1/share/elpt_2023_000518/15_maqun/FHIST_f19_br.cam.h0.2008-09-09-00000.nc"
hires_path = r"/data1/share/elpt_2023_000518/15_maqun/DemHiRes.nc"
out_path = r"/data1/share/elpt_2023_000518/15_maqun/maqun_HiRes.nc"

# ==== Step 1. 打开CESM数据 ====
with open(cesm_path, 'rb') as f:
    ds_cesm = xr.open_dataset(f)

ps = ds_cesm['TS'].squeeze()   # (lat, lon)
lat_cesm = ds_cesm['lat']
lon_cesm = ds_cesm['lon']

# ==== Step 2. 打开高分辨率网格 ====
with open(hires_path, 'rb') as f:
    ds_hi = xr.open_dataset(f)

lat_hi = ds_hi['lat']
lon_hi = ds_hi['lon']

# ==== Step 3. 进行插值 ====
# 构造二维网格
lon_cesm_2d, lat_cesm_2d = np.meshgrid(lon_cesm, lat_cesm)
lon_hi_2d, lat_hi_2d = np.meshgrid(lon_hi, lat_hi)

points = np.column_stack((lon_cesm_2d.ravel(), lat_cesm_2d.ravel()))
values = ps.values.ravel()
points_target = np.column_stack((lon_hi_2d.ravel(), lat_hi_2d.ravel()))

# 双线性插值
ps_interp = griddata(points, values, points_target, method='linear')
ps_interp = ps_interp.reshape(lat_hi.size, lon_hi.size)

# ==== Step 4. 生成新数据集 ====
ds_out = xr.Dataset(
    {
        "TS": (["lat", "lon"], ps_interp)
    },
    coords={
        "lat": lat_hi,
        "lon": lon_hi
    }
)

# ==== Step 5. 保存（注意写入中文路径时用 encode 方式）====
with open(out_path.encode('utf-8'), "wb") as f:
    ds_out.to_netcdf(f)

print("✅ 插值完成！结果已保存到：", out_path)
