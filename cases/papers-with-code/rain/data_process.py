import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.cluster import DBSCAN
import os
import glob


def read_observation_data(file_path,targe_year=2021,target_month=7):
    """
    读取观测站点的降水数据,Read data file to dataframe
    """
    # 尝试不同的编码格式读取原始文本
    encodings = ['gbk', 'gb2312', 'latin1', 'utf-8']

    # 使用二进制模式读取文件，然后尝试不同的编码解码
    with open(file_path, 'rb') as f:
        content_bytes = f.read()

    content = None
    for encoding in encodings:
        try:
            content = content_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise Exception("无法读取文件，请检查文件编码")

    # 按空格分割内容，得到所有记录（包括标题）
    records = content.strip().split(' ')

    # 第一个记录应该是标题
    header = records[0].split(',')

    # 预分配内存，提高效率
    data_dict = {col: [] for col in header}

    # 解析数据记录
    for record in records[1:]:
        if record.strip():  # 跳过空记录
            fields = record.split(',')
            if len(fields) == len(header):
                for i, col in enumerate(header):
                    data_dict[col].append(fields[i])

    # 创建DataFrame
    df = pd.DataFrame(data_dict)

    # 一次性转换数据类型，减少循环
    numeric_cols = ['Year', 'Mon', 'Day', 'Hour', 'Lat', 'Lon', 'PRE_1h']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 筛选2021年7月的数据
    df = df[(df['Year'] == targe_year) & (df['Mon'] == target_month)]

    # 按站点位置和日期分组，计算日降水量
    df['date'] = pd.to_datetime(df[['Year', 'Mon', 'Day']].astype(str).agg('-'.join, axis=1))
    daily_rainfall = df.groupby(['Lon', 'Lat', 'date'])['PRE_1h'].sum().reset_index()

    # 重命名列以匹配后续处理
    daily_rainfall.rename(columns={'Lon': 'lon', 'Lat': 'lat', 'PRE_1h': 'rainfall'}, inplace=True)

    return daily_rainfall


def interpolate_to_grid(daily_rainfall, lon_min=110, lon_max=114, lat_min=32, lat_max=36, resolution=0.125):
    """将站点数据插值到规则网格"""
    # 创建规则网格
    grid_lon = np.arange(lon_min, lon_max + resolution, resolution)
    grid_lat = np.arange(lat_min, lat_max + resolution, resolution)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    # 按日期分组进行插值
    dates = daily_rainfall['date'].unique()
    grid_data = {}

    for date in dates:
        day_data = daily_rainfall[daily_rainfall['date'] == date]
        points = day_data[['lon', 'lat']].values
        values = day_data['rainfall'].values

        # 使用线性插值方法
        grid_values = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='linear')

        # 存储结果
        grid_data[date] = {
            'lon': grid_lon,
            'lat': grid_lat,
            'rainfall': grid_values
        }

    return grid_data


def read_model_data(folder_path):
    """读取模式数据"""
    model_data = {}

    # 获取所有模式数据文件
    file_list = glob.glob(os.path.join(folder_path, '*.txt'))
    
    # 记录成功和失败的文件数
    success_count = 0
    failure_count = 0

    for file_path in file_list:
        # 从文件名提取日期
        file_name = os.path.basename(file_path)
        date_str = file_name.split('.')[0]  # 文件名格式为 YYMMDD.txt
        date = pd.to_datetime('20' + date_str, format='%Y%m%d')  # 假设年份是21世纪

        try:
            # 读取模式数据
            with open(file_path, 'r', errors='replace') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                print(f"错误：文件 {file_name} 内容不足")
                failure_count += 1
                continue
                
            # 尝试自动检测文件格式和数据结构
            header_line = lines[0].strip()
            
            # 默认值 - 研究区域的标准范围
            lon_min, lon_max = 105.0, 125.0
            lat_max, lat_min = 40.0, 25.0
            nx, ny = 161, 121
            
            # 检查是否是带有"diamond"前缀的特殊格式
            if header_line.startswith("diamond"):
                # 这是一种特殊格式，例如：
                # diamond 4  21070820.036__ECMWF_HR_RAIN24(units:mm)    21     7     8    20    36  9999     0.125    -0.125   105.000   125.000    40.000    25.000   161   121    10.000     0.000   100.000     1.000     0.000
                header_parts = header_line.split()
                
                # 检查是否有足够的部分来提取参数
                if len(header_parts) >= 18:
                    try:
                        # 从固定位置提取参数
                        lon_min = float(header_parts[12])
                        lon_max = float(header_parts[13])
                        lat_max = float(header_parts[14])
                        lat_min = float(header_parts[15])
                        nx = int(float(header_parts[16]))
                        ny = int(float(header_parts[17]))
                        print(f"从diamond格式头信息中提取参数: 经度范围 {lon_min}-{lon_max}, 纬度范围 {lat_max}-{lat_min}, 网格大小 {nx}x{ny}")
                    except (ValueError, IndexError) as e:
                        print(f"警告：无法从diamond格式头信息中提取参数: {e}")
                else:
                    print(f"警告：diamond格式头信息不完整，使用默认值")
            else:
                # 尝试标准格式解析
                header_parts = header_line.split()
                try:
                    # 提取所有可能是数字的部分
                    numbers = []
                    for part in header_parts:
                        try:
                            numbers.append(float(part))
                        except ValueError:
                            pass
                    
                    # 如果找到至少6个数字，假设它们是我们需要的参数
                    if len(numbers) >= 6:
                        # 标准格式: 0.125 -0.125 105.000 125.000 40.000 25.000 161 121 10.000 0.000 100.000 1.000 0.000
                        if len(numbers) >= 8:
                            lon_min, lon_max = numbers[2], numbers[3]
                            lat_max, lat_min = numbers[4], numbers[5]
                            nx, ny = int(float(numbers[6])), int(float(numbers[7]))
                            print(f"从标准格式头信息中提取参数: 经度范围 {lon_min}-{lon_max}, 纬度范围 {lat_max}-{lat_min}, 网格大小 {nx}x{ny}")
                        else:
                            print(f"警告：标准格式头信息不完整，使用默认值")
                except Exception as e:
                    print(f"警告：文件 {file_name} 头信息解析失败: {e}")
            
            # 从数据部分读取降水值
            data_lines = lines[1:]
            data_values = []
            
            for line in data_lines:
                try:
                    values = [float(val) for val in line.strip().split()]
                    data_values.extend(values)
                except ValueError:
                    continue
            
            total_points = len(data_values)
            
            # 验证网格大小是否有效
            if nx <= 1 or ny <= 1:
                print(f"警告：文件 {file_name} 的网格大小无效 ({nx}x{ny})，将使用推断的网格大小")
                nx, ny = 0, 0  # 重置为无效值，强制推断网格大小
            
            # 如果数据点数与网格大小不匹配，尝试推断网格大小
            if total_points != nx * ny:
                print(f"警告：文件 {file_name} 的数据点数 ({total_points}) 与网格大小 ({nx}x{ny}={nx*ny}) 不匹配")
                
                # 尝试找到合适的网格大小
                possible_dimensions = []
                for possible_nx in range(10, 500):
                    if total_points % possible_nx == 0:
                        possible_ny = total_points // possible_nx
                        possible_dimensions.append((possible_nx, possible_ny))
                
                if possible_dimensions:
                    # 选择最接近正方形的网格尺寸
                    best_ratio = float('inf')
                    best_dim = None
                    for dim in possible_dimensions:
                        ratio = max(dim[0]/dim[1], dim[1]/dim[0])
                        if ratio < best_ratio:
                            best_ratio = ratio
                            best_dim = dim
                    
                    nx, ny = best_dim
                    print(f"推断网格大小为: {nx}x{ny}")
                else:
                    print(f"错误：无法为文件 {file_name} 推断合适的网格大小")
                    failure_count += 1
                    continue
            
            # 重新计算经纬度网格
            lons = np.linspace(lon_min, lon_max, nx)
            lats = np.linspace(lat_max, lat_min, ny)
            
            # 将一维数据重塑为二维网格
            try:
                rainfall_grid = np.array(data_values).reshape(ny, nx)
            except ValueError as e:
                print(f"错误：文件 {file_name} 无法重塑数据: {e}")
                failure_count += 1
                continue

            # 筛选研究区域内的数据（32-36N，110-114E）
            mask_lon = (lons >= 110) & (lons <= 114)
            mask_lat = (lats >= 32) & (lats <= 36)

            if not any(mask_lon) or not any(mask_lat):
                print(f"警告：文件 {file_name} 的数据不在研究区域内")
                failure_count += 1
                continue

            filtered_lons = lons[mask_lon]
            filtered_lats = lats[mask_lat]
            filtered_rainfall = rainfall_grid[np.ix_(mask_lat, mask_lon)]

            model_data[date] = {
                'lon': filtered_lons,
                'lat': filtered_lats,
                'rainfall': filtered_rainfall
            }
            success_count += 1
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
            failure_count += 1
            continue

    print(f"成功读取 {success_count} 个文件，失败 {failure_count} 个文件")
    return model_data


# def identify_rainfall_clusters(grid_data, threshold=25, min_size=8):
#     """识别降水目标（大于阈值且连续覆盖范围大于1°×1°的降水区域）"""
#     clusters_by_date = {}
#
#     for date, data in grid_data.items():
#         rainfall = data['rainfall']
#
#         # 创建二值掩码：大于阈值的区域为1，其他为0
#         mask = np.zeros_like(rainfall, dtype=int)
#         mask[rainfall >= threshold] = 1
#
#         # 如果没有降水超过阈值，跳过
#         if np.sum(mask) == 0:
#             continue
#
#         # 将二维数组展平为点集合
#         points = []
#         for i in range(mask.shape[0]):
#             for j in range(mask.shape[1]):
#                 if mask[i, j] == 1:
#                     points.append([j, i])  # 注意：j对应经度，i对应纬度
#
#         if not points:
#             continue
#
#         points = np.array(points)
#
#         # 使用DBSCAN进行聚类
#         # eps参数设置为1.5倍的网格分辨率，确保相邻格点被归为同一类
#         # min_samples设置为min_size，确保聚类大小至少为1°×1°
#         db = DBSCAN(eps=1.5, min_samples=min_size).fit(points)
#
#         labels = db.labels_
#
#         # 提取聚类结果
#         clusters = {}
#         for i, label in enumerate(labels):
#             if label == -1:  # 噪声点
#                 continue
#
#             if label not in clusters:
#                 clusters[label] = []
#
#             clusters[label].append(points[i])
#
#         clusters_by_date[date] = clusters
#
#     return clusters_by_date
