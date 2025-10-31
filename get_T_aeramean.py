import numpy as np
import xarray as xr

# ------------------------------
# 1. 读取 HiRes 网格数据
# ------------------------------
ncfile = r"D:\data\temp\maqun_HiRes.nc"
# print(os.path.exists(ncfile))
# with open(ncfile, 'rb') as f:
ds = xr.open_dataset(ncfile)

# 假设变量名为 PS（地表温度），经纬度为 lat、lon
temp = ds["TS"].values
lat = ds["lat"]
lon = ds["lon"]

def Area_Mean(data, lat, lon):
    '''
    by XiaoMaFenJu
    不适合数据含nan
    data: 要进行区域加权平均的变量，支持2、3维  2D: [lat, lon]  3D：[time, lat, lon]
    lat: data2D对应的纬度 1D
    lon: data2D对应的经度 1D
    '''

    if data.ndim == 2:
        y_weight2D = abs(np.cos(lat*np.pi/180))
        weight2D = np.expand_dims(y_weight2D, 1).repeat(len(lon), axis=1)
        # print(weight2D)
        new_data = np.average(data, weights=weight2D)
        return new_data
    elif data.ndim == 3:
        y_weight2D = abs(np.cos(lat*np.pi/180))
        weight2D = np.expand_dims(y_weight2D, 1).repeat(len(lon), axis=1)
        weight3D = np.expand_dims(weight2D,0).repeat(len(data[:,0,0]),axis=0)
        new_data = np.average(data, weights=weight3D, axis=(-1, -2))
        return new_data
    else:
        print('输入数据非2&3维')
        pass
    pass

t_mean = Area_Mean(temp, lat, lon)
print(f"{t_mean:.5} K")
print(f"{t_mean-273.15:.3} ℃")