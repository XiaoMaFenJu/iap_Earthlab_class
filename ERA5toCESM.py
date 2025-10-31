#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpolate_ERA5_to_CESM_init_scipy.py

功能：
1) 读取 ERA5 GRIB 数据 (t,q,u,v,sp)，仅取 time=0
2) 使用 scipy 双线性插值将 ERA5 水平插值到 CESM 网格
3) 使用 log(p) 线性插值将 ERA5 等压层垂直插值到 CESM 混合层
4) 替换 CESM 初始场中对应变量并保存新文件
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

cesm_init = "/data1/share/elpt_2023_000518/15_maqun/20080907/FHIST_f19.cam.r.2008-09-07-00000.nc"
out_nc = "/data1/share/elpt_2023_000518/15_maqun/cesm2.1.4/run/FHIST_f19/run/FHIST_f19.cam.r.2008-09-07-00000.nc"

# ----------------- 读取 CESM 初始场 -----------------
ds_cesm = xr.open_dataset(cesm_init, decode_times=False)

lat = ds_cesm["lat"].values
lon = ds_cesm["lon"].values

hyam = ds_cesm["hyam"].values
hybm = ds_cesm["hybm"].values
PS_cesm = ds_cesm["PS"].values  # Pa
p0 = 100000.0

# 混合层压力：根据 hyam/hybm 范围确定系数单位
# hyam 在 3.6–922.5 → 单位近似 hPa，所以乘 100
# p_target = hyam[:, None, None] * 100.0 + hybm[:, None, None] * PS_cesm[None, :, :]
# 主格点（T, Q, PS, VS用）
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

# 确保经度都在 0-360
def to360(lon_arr):
    lon = np.mod(lon_arr, 360)
    return lon

era_lons = to360(era_lons)
lon360 = to360(lon)

# ----------------- 定义水平双线性插值函数 -----------------
def bilinear_interp_scipy(data, src_lats, src_lons, tgt_lats, tgt_lons):
    """
    data: shape (..., nlat, nlon)
    返回: 与目标网格匹配的新数组 (..., n_tlat, n_tlon)
    """
    # 注意：RegularGridInterpolator 的坐标必须升序
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

# 3D版本（用于 t,q,u,v）
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
u_on_cesm = bilinear_interp_3d(era_u.values, era_lats, era_lons, lat, lon)
print("水平重网格 V...")
v_on_cesm = bilinear_interp_3d(era_v.values, era_lats, era_lons, lat, lon)

# ----------------- 垂直插值（log(p)空间） -----------------
def vertical_interp_logp(era_val, era_p, target_p):
    """
    era_val: shape (nlev_era, ny, nx)
    era_p: 1D (Pa)
    target_p: shape (nlev_cesm, ny, nx)
    返回: shape (nlev_cesm, ny, nx)
    """
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
# U格点 (slat, lon)
# 需要插值PS到slat网格
PS_on_slat = bilinear_interp_scipy(PS_cesm, lat, lon, lat, lon)
p_target_u = hyam[:, None, None] * 100.0 + hybm[:, None, None] * PS_on_slat[None, :, :]

# V格点 (lat, slon)
PS_on_slon = bilinear_interp_scipy(PS_cesm, lat, lon, lat, lon)
p_target_v = hyam[:, None, None] * 100.0 + hybm[:, None, None] * PS_on_slon[None, :, :]

print("垂直插值 T...")
t_vert = vertical_interp_logp(t_on_cesm, era_plev, p_target)
print("垂直插值 Q...")
q_vert = vertical_interp_logp(q_on_cesm, era_plev, p_target)
print("垂直插值 U...")
u_vert = vertical_interp_logp(u_on_cesm, era_plev, p_target_u)
print("垂直插值 V...")
v_vert = vertical_interp_logp(v_on_cesm, era_plev, p_target_v)

# ----------------- 写入到 CESM 初始场 -----------------
ds_out = ds_cesm.copy(deep=True)

# time_val = ds_cesm["time"].isel(time=0).values

ds_out["PS"].values = sp_on_cesm.astype(ds_out["PS"].dtype)
ds_out["T_TTEND"][:, :, :] = t_vert.astype(ds_out['T_TTEND'].dtype)
ds_out["Q"][:, :, :] = q_vert.astype(ds_out['Q'].dtype)
ds_out["U"][:, :, :] = u_vert.astype(ds_out['U'].dtype)
ds_out["V"][:, :, :] = v_vert.astype(ds_out['V'].dtype)

# ds_out["date"].values = [20080907]
# # ds_out["nbdate"].values = 20070907
# ds_out["ndcur"].values = [10842]
# ds_out["nsteph"].values = [520416]
#
# ds_out = ds_out.assign_coords(time=[10842.0])
# ds_out["time_bnds"].values = [[10842.0, 10842.0]]
# ds_out["time"].attrs["long_name"] = "time"
# ds_out["time"].attrs["units"] = "days since 1979-01-01 00:00:00"
# ds_out["time"].attrs["calendar"] = "noleap"
# ds_out["time"].attrs["bounds"] = "time_bnds"

# ds_out.attrs["history"] = (
#     "ERA5 (time=0) bilinearly regridded (scipy) and vertically interpolated (log-p) "
#     "to CESM hybrid levels. hyam/hybm interpreted as A(hPa)/B. "
# )

print("保存输出文件到:", out_nc)
comp = dict(zlib=True, complevel=4)
encoding = {v: comp for v in ds_out.data_vars}
ds_out.to_netcdf(out_nc, format="NETCDF4", encoding=encoding)
print("完成。输出文件:", out_nc)
