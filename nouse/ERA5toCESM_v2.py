#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpolate_ERA5_to_CESM_subset.py

功能：
1) 读取 ERA5 GRIB 数据 (t,q,u,v,sp)，仅取 time=0
2) 插值到 CESM 网格
3) 生成只包含这5个变量的新nc文件，时间设为2008年9月7日00:00
"""

import os
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# ----------------- 输入输出路径 -----------------
era_dir = "/data1/share/elpt_2023_000518/15_maqun/20080907"
era_files = {
    "q": os.path.join(era_dir, "era5_hourly_specific_humidity_20080907_global.grib"),
    "sp": os.path.join(era_dir, "era5_hourly_surface_pressure_20080907_global.grib"),
    "t": os.path.join(era_dir, "era5_hourly_temperature_20080907_global.grib"),
    "u": os.path.join(era_dir, "era5_hourly_u_component_of_wind_20080907_global.grib"),
    "v": os.path.join(era_dir, "era5_hourly_v_component_of_wind_20080907_global.grib"),
}

cesm_init = "/data1/share/elpt_2023_000518/15_maqun/20080907/b.e20.B1850.f19_g17.release_cesm2_1_0.020.cam.i.0301-01-01-00000.nc"
out_nc = "/data1/share/elpt_2023_000518/15_maqun/20080907/B1850.f19_g17.cam.i.20080907-00000.nc"

# ----------------- 读取 CESM 初始场（主要获取网格信息） -----------------
ds_cesm = xr.open_dataset(cesm_init, decode_times=False)

lat = ds_cesm["lat"].values
lon = ds_cesm["lon"].values

slat = ds_cesm["slat"].values
slon = ds_cesm["slon"].values

hyam = ds_cesm["hyam"].values
hybm = ds_cesm["hybm"].values
PS_cesm = ds_cesm["PS"].isel(time=0).values  # Pa
p0 = 100000.0

# 混合层压力计算
p_target = hyam[:, None, None] * 100.0 + hybm[:, None, None] * PS_cesm[None, :, :]

# ----------------- 读取 ERA5 数据 -----------------
def read_era5_var(path, varname):
    ds = xr.open_dataset(path, engine="cfgrib")
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    da = ds[varname].isel(time=0)
    da = da.rename({lat_name: "lat", lon_name: "lon"})
    return da

era_sp = read_era5_var(era_files["sp"], "sp")
era_t = read_era5_var(era_files["t"], "t")
era_q = read_era5_var(era_files["q"], "q")
era_u = read_era5_var(era_files["u"], "u")
era_v = read_era5_var(era_files["v"], "v")

era_lats = era_sp["lat"].values
era_lons = era_sp["lon"].values
era_plev = era_t["isobaricInhPa"].values * 100.0  # hPa → Pa

# 确保经度在 0-360 范围
def to360(lon_arr):
    return np.mod(lon_arr, 360)

era_lons = to360(era_lons)
lon360 = to360(lon)
slon360 = to360(slon)
# ----------------- 水平双线性插值函数 -----------------
def bilinear_interp_scipy(data, src_lats, src_lons, tgt_lats, tgt_lons):
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        data = data[..., ::-1, :]
    if src_lons[0] > src_lons[-1]:
        src_lons = src_lons[::-1]
        data = data[..., :, ::-1]

    interp_func = RegularGridInterpolator(
        (src_lats, src_lons), data, bounds_error=False, fill_value=np.nan
    )

    tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lons, tgt_lats)
    tgt_points = np.array([tgt_lat2d.ravel(), tgt_lon2d.ravel()]).T
    out = interp_func(tgt_points).reshape((*tgt_lat2d.shape,))
    return out

def bilinear_interp_3d(data3d, src_lats, src_lons, tgt_lats, tgt_lons):
    nlev = data3d.shape[0]
    out = np.empty((nlev, len(tgt_lats), len(tgt_lons)))
    for k in range(nlev):
        out[k] = bilinear_interp_scipy(data3d[k, :, :], src_lats, src_lons, tgt_lats, tgt_lons)
    return out

# ----------------- 水平重网格 -----------------
print("水平重网格 SP...")
sp_on_cesm = bilinear_interp_scipy(era_sp.values, era_lats, era_lons, lat, lon)

print("水平重网格 T...")
t_on_cesm = bilinear_interp_3d(era_t.values, era_lats, era_lons, lat, lon)
print("水平重网格 Q...")
q_on_cesm = bilinear_interp_3d(era_q.values, era_lats, era_lons, lat, lon)
print("水平重网格 U...")
u_on_cesm = bilinear_interp_3d(era_u.values, era_lats, era_lons, slat, lon)
print("水平重网格 V...")
v_on_cesm = bilinear_interp_3d(era_v.values, era_lats, era_lons, lat, slon)

# ----------------- 垂直插值（log(p)空间） -----------------
def vertical_interp_logp(era_val, era_p, target_p):
    logp_era = np.log(era_p)
    sort_idx = np.argsort(logp_era)
    logp_era = logp_era[sort_idx]
    era_val = era_val[sort_idx, :, :]
    nlev_tgt = target_p.shape[0]
    out = np.empty_like(target_p)
    for i in range(target_p.shape[1]):  # lat
        for j in range(target_p.shape[2]):  # lon
            prof = era_val[:, i, j]
            if np.all(np.isnan(prof)):
                out[:, i, j] = np.nan
                continue
            logp_tgt = np.log(target_p[:, i, j])
            good = np.isfinite(prof)
            if good.sum() < 2:
                out[:, i, j] = np.nan
                continue
            out[:, i, j] = np.interp(logp_tgt, logp_era[good], prof[good],
                                     left=prof[good][0], right=prof[good][-1])
    return out

# U/V格点压力计算
PS_on_slat = bilinear_interp_scipy(PS_cesm, lat, lon, slat, lon)
p_target_u = hyam[:, None, None] * 100.0 + hybm[:, None, None] * PS_on_slat[None, :, :]

PS_on_slon = bilinear_interp_scipy(PS_cesm, lat, lon, lat, slon)
p_target_v = hyam[:, None, None] * 100.0 + hybm[:, None, None] * PS_on_slon[None, :, :]

print("垂直插值 T...")
t_vert = vertical_interp_logp(t_on_cesm, era_plev, p_target)
print("垂直插值 Q...")
q_vert = vertical_interp_logp(q_on_cesm, era_plev, p_target)
print("垂直插值 U...")
u_vert = vertical_interp_logp(u_on_cesm, era_plev, p_target_u)
print("垂直插值 V...")
v_vert = vertical_interp_logp(v_on_cesm, era_plev, p_target_v)

# ----------------- 创建只包含5个变量的新数据集 -----------------
# 创建时间坐标（2008-09-07 00:00:00）
time = xr.DataArray(
    [10842.0],
    dims='time',
    attrs={
        'long_name': 'time',
        'units': 'days since 1979-01-01 00:00:00',
        'calendar': 'noleap',
        'bounds': 'time_bnds'
    }
)

# 创建时间边界
time_bnds = xr.DataArray(
    [[10842.0, 10842.0]],
    dims=['time', 'nbnds'],
    attrs={'long_name': 'time interval boundaries'}
)

# 创建新数据集
ds_out = xr.Dataset(
    coords={
        'lat': (['lat'], lat, ds_cesm['lat'].attrs),
        'lon': (['lon'], lon, ds_cesm['lon'].attrs),
        'slat': (['slat'], slat, ds_cesm['slat'].attrs),
        'slon': (['slon'], slon, ds_cesm['slon'].attrs),
        'lev': (['lev'], ds_cesm['lev'].values, ds_cesm['lev'].attrs),
        'time': time,
        'time_bnds': time_bnds
    }
)

sp_on_cesm = sp_on_cesm[np.newaxis,:]
# 添加变量
ds_out['PS'] = xr.DataArray(
    sp_on_cesm.astype(ds_cesm["PS"].dtype),
    dims=['time','lat', 'lon'],
    attrs=ds_cesm['PS'].attrs
)

t_vert = t_vert[np.newaxis,:]
ds_out['T'] = xr.DataArray(
    t_vert.astype(ds_cesm['T'].dtype),
    dims=['time','lev', 'lat', 'lon'],
    attrs=ds_cesm['T'].attrs
)

q_vert = q_vert[np.newaxis,:]
ds_out['Q'] = xr.DataArray(
    q_vert.astype(ds_cesm['Q'].dtype),
    dims=['time','lev', 'lat', 'lon'],
    attrs=ds_cesm['Q'].attrs
)

u_vert = u_vert[np.newaxis,:]
ds_out['US'] = xr.DataArray(
    u_vert.astype(ds_cesm['US'].dtype),
    dims=['time','lev', 'slat', 'lon'],
    attrs=ds_cesm['US'].attrs
)

v_vert = v_vert[np.newaxis,:]
ds_out['VS'] = xr.DataArray(
    v_vert.astype(ds_cesm['VS'].dtype),
    dims=['time','lev', 'lat', 'slon'],
    attrs=ds_cesm['VS'].attrs
)

# # 添加全局属性
# ds_out.attrs = {
#     'title': 'ERA5 data interpolated to CESM grid',
#     'source': 'ERA5 hourly data (t,q,u,v,sp) interpolated using bilinear (horizontal) and log-p (vertical) methods',
#     'history': 'Created: ' + os.popen('date').read().strip(),
#     'creation_date': os.popen('date').read().strip()
# }

# ----------------- 保存输出文件 -----------------
print("保存输出文件到:", out_nc)
comp = dict(zlib=True, complevel=4)
encoding = {v: comp for v in ds_out.data_vars}
ds_out.to_netcdf(out_nc, format="NETCDF4", encoding=encoding)
print("完成。输出文件:", out_nc)