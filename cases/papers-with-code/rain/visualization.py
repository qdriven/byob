import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import os

# 在文件顶部添加或修改导入语句
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import os
from cartopy.io import shapereader
from cartopy.feature import ShapelyFeature

# 尝试导入 geopandas，如果不可用则提供警告
# try:
#     import geopandas as gpd

#     HAS_GPD = True
# except ImportError:
#     HAS_GPD = False
#     print("警告: geopandas 未安装，将不会显示市级区划")

# 在使用 gpd 的函数中添加检查

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 在文件顶部的字体设置部分
# 确保使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'PingFang SC', 'Heiti SC', 'Microsoft YaHei',
                                   'SimHei']  # macOS 和 Windows 常见中文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 修改 plot_rainfall_contour 函数，移除区域信息相关代码
# 修改 plot_rainfall_contour 函数中的颜色设置部分
def plot_rainfall_contour(grid_data, date, save_path=None, title=None, city_data=None, check_coverage=False):
    """
    绘制单日降水等值图

    参数:
    grid_data: 网格化的观测数据
    date: 日期
    save_path: 保存路径
    title: 标题
    city_data: 不再使用，保留参数以兼容现有代码
    check_coverage: 不再使用，保留参数以兼容现有代码
    """
    lon = grid_data['lon']
    lat = grid_data['lat']
    rainfall = grid_data['rainfall']

    # 创建画布和地图投影
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 创建经纬度网格
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)

    # 设置降水等级和颜色 - 与提供的图片一致
    levels = [0.1, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]
    colors = ['#E8F5E9', '#A5D6A7', '#4CAF50', '#81D4FA', '#2196F3', '#E040FB', '#9C27B0']
    cmap = LinearSegmentedColormap.from_list('rainfall', colors, N=len(levels))
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # 绘制填充等值图
    cf = ax.contourf(lon_mesh, lat_mesh, rainfall, levels=levels,
                     cmap=cmap, norm=norm, extend='both', zorder=5)

    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=10)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6, zorder=10)

    # 添加简单的经纬度网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='-', zorder=15)
    gl.top_labels = False
    gl.right_labels = False

    # 设置坐标轴范围和标签
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 添加颜色条 - 与提供的图片一致，放在右侧
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=20)
    cbar.set_label('mm')

    # 设置标题 - 与提供的图片一致的简洁风格
    if title is None:
        title = f'Daily rainfall on {date.strftime("%d")}'
    ax.set_title(title)

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# 修改 plot_observation_vs_model_improved 函数，确保实现所需功能
# 修改 plot_observation_vs_model_improved 函数中的颜色设置部分
def plot_observation_vs_model_improved(obs_data, model_data, date, save_path=None, title=None):
    """
    绘制观测数据与模式数据对比图
    观测数据用填充色，模拟数据用等值线，等值线标注数值

    参数:
    obs_data: 观测数据
    model_data: 模式数据
    date: 日期
    save_path: 保存路径
    title: 图像标题
    """
    # 创建图形和坐标系
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 设置地图范围
    lon_min, lon_max = 110, 114
    lat_min, lat_max = 32, 36
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # 添加地图要素，保持简洁风格
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)

    # 获取观测数据
    obs = obs_data[date]
    obs_lons, obs_lats = np.meshgrid(obs['lon'], obs['lat'])
    obs_rainfall = obs['rainfall']

    # 获取模式数据
    model = model_data[date]
    model_lons, model_lats = np.meshgrid(model['lon'], model['lat'])
    model_rainfall = model['rainfall']

    # 设置降水等级和颜色
    levels = [0.1, 10, 25, 50, 100, 250, 500]
    colors = ['#E8F5E9', '#A5D6A7', '#4CAF50', '#81D4FA', '#2196F3', '#E040FB', '#9C27B0']
    cmap = LinearSegmentedColormap.from_list('rainfall', colors, N=len(levels))
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # 绘制观测数据（填充色）
    cf = ax.contourf(obs_lons, obs_lats, obs_rainfall, levels=levels,
                     cmap=cmap, norm=norm, extend='both')

    # 绘制模式数据（等值线）- 使用指定的等值线级别
    contour_levels = [0, 10, 25, 50, 100, 250]
    cs = ax.contour(model_lons, model_lats, model_rainfall, levels=contour_levels,
                    colors='black', linewidths=0.8)

    # 在等值线上标注数值
    ax.clabel(cs, inline=True, fontsize=8, fmt='%g')

    # 添加简单的经纬度网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='-')
    gl.top_labels = False
    gl.right_labels = False

    # 设置坐标轴范围和标签
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')

    # 添加颜色条
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.05, aspect=20)
    cbar.set_label('降水量 (mm)')

    # 添加标题
    if title is None:
        title = f'观测与模拟降水对比 - {date.strftime("%Y-%m-%d")}'
    ax.set_title(title)

    # 添加图例说明
    ax.text(0.02, 0.02, '填充色: 观测数据\n等值线: 模拟数据',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.7))

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _plot_clusters(data, clusters, date, data_type, save_path=None, city_data=None):
    """
    绘制聚类结果

    参数:
    data: 网格化的数据
    clusters: 聚类结果
    date: 日期
    data_type: 数据类型（观测数据或模式数据）
    save_path: 保存路径
    city_data: 不再使用，保留参数以兼容现有代码
    """
    lon = data['lon']
    lat = data['lat']
    rainfall = data['rainfall']

    # 创建画布和地图投影
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 创建经纬度网格
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)

    # 设置降水等级和颜色 - 与提供的图片一致
    levels = [0.1, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]
    colors = ['#E8F5E9', '#A5D6A7', '#4CAF50', '#81D4FA', '#2196F3', '#E040FB', '#9C27B0']
    cmap = LinearSegmentedColormap.from_list('rainfall', colors, N=len(levels))
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # 绘制填充等值图
    cf = ax.contourf(lon_mesh, lat_mesh, rainfall, levels=levels,
                     cmap=cmap, norm=norm, extend='both', zorder=5)

    # 添加地图要素
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=10)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6, zorder=10)

    # 为每个聚类绘制边界
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, (label, points) in enumerate(clusters.items()):
        color = cluster_colors[i % len(cluster_colors)]
        points = np.array(points)

        # 将点的索引转换为经纬度坐标
        x_coords = lon[points[:, 0]]
        y_coords = lat[points[:, 1]]

        # 绘制聚类边界
        ax.scatter(x_coords, y_coords, color=color, s=10, alpha=0.7, zorder=20,
                   label=f'目标 {label + 1}')

        # 计算聚类中心
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)

        # 计算聚类区域的平均降水量
        cluster_rainfall = []
        for point in points:
            cluster_rainfall.append(rainfall[point[1], point[0]])
        avg_rainfall = np.mean(cluster_rainfall)

        # 在聚类中心添加标签
        ax.text(center_x, center_y, f'{label + 1}: {avg_rainfall:.1f}mm',
                fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                zorder=25)

    # 添加简单的经纬度网格线
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='-', zorder=15)
    gl.top_labels = False
    gl.right_labels = False

    # 设置坐标轴范围和标签
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')

    # 添加颜色条
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=20)
    cbar.set_label('降水量 (mm)')

    # 设置标题
    title = f'{data_type}降水目标识别 - {date.strftime("%Y-%m-%d")}'
    ax.set_title(title)

    # 添加图例
    if clusters:
        ax.legend(loc='lower left', fontsize=8)

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def identify_and_plot_obs_clusters(grid_data, threshold=25, min_size=8, save_path=None, city_data=None):
    """
    识别并绘制观测数据中的降水目标（大于阈值的区域）

    参数:
    grid_data: 网格化的观测数据
    threshold: 降水阈值，默认25mm
    min_size: 最小聚类大小
    save_path: 保存路径
    city_data: 不再使用，保留参数以兼容现有代码
    """
    from sklearn.cluster import DBSCAN

    # 创建结果字典
    clusters_by_date = {}

    for date, data in grid_data.items():
        rainfall = data['rainfall']

        # 创建二值掩码：大于阈值的区域为1，其他为0
        mask = np.zeros_like(rainfall, dtype=int)
        mask[rainfall >= threshold] = 1

        # 如果没有降水超过阈值，跳过
        if np.sum(mask) == 0:
            print(f"{date.strftime('%Y-%m-%d')} 没有降水超过 {threshold}mm 的区域")
            continue

        # 将二维数组展平为点集合
        points = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 1:
                    points.append([j, i])  # 注意：j对应经度，i对应纬度

        if not points:
            continue

        points = np.array(points)

        # 使用DBSCAN进行聚类
        db = DBSCAN(eps=1.5, min_samples=min_size).fit(points)

        labels = db.labels_

        # 提取聚类结果
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # 噪声点
                continue

            if label not in clusters:
                clusters[label] = []

            clusters[label].append(points[i])

        clusters_by_date[date] = clusters

        # 绘制聚类结果
        if save_path:
            date_save_path = save_path.replace('.png', f'_{date.strftime("%Y%m%d")}.png')
            _plot_clusters(data, clusters, date, "观测数据", date_save_path)

            # 输出聚类信息
            print(f"{date.strftime('%Y-%m-%d')} 观测数据中识别到 {len(clusters)} 个降水目标")
            for label, points in clusters.items():
                # 计算该聚类的平均降水量
                cluster_rainfall = []
                for point in points:
                    cluster_rainfall.append(rainfall[point[1], point[0]])
                avg_rainfall = np.mean(cluster_rainfall)
                max_rainfall = np.max(cluster_rainfall)

                print(
                    f"  目标 {label + 1}: 包含 {len(points)} 个格点, 平均降水量 {avg_rainfall:.1f}mm, 最大降水量 {max_rainfall:.1f}mm")

    return clusters_by_date


def identify_and_plot_model_clusters(model_data, threshold=25, min_size=8, save_path=None, city_data=None):
    """
    识别并绘制模式数据中的降水目标（大于阈值的区域）

    参数:
    model_data: 模式数据
    threshold: 降水阈值，默认25mm
    min_size: 最小聚类大小
    save_path: 保存路径
    city_data: 不再使用，保留参数以兼容现有代码
    """
    from sklearn.cluster import DBSCAN

    # 创建结果字典
    clusters_by_date = {}

    for date, data in model_data.items():
        rainfall = data['rainfall']

        # 创建二值掩码：大于阈值的区域为1，其他为0
        mask = np.zeros_like(rainfall, dtype=int)
        mask[rainfall >= threshold] = 1

        # 如果没有降水超过阈值，跳过
        if np.sum(mask) == 0:
            print(f"{date.strftime('%Y-%m-%d')} 没有降水超过 {threshold}mm 的区域")
            continue

        # 将二维数组展平为点集合
        points = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 1:
                    points.append([j, i])  # 注意：j对应经度，i对应纬度

        if not points:
            continue

        points = np.array(points)

        # 使用DBSCAN进行聚类
        db = DBSCAN(eps=1.5, min_samples=min_size).fit(points)

        labels = db.labels_

        # 提取聚类结果
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # 噪声点
                continue

            if label not in clusters:
                clusters[label] = []

            clusters[label].append(points[i])

        clusters_by_date[date] = clusters

        # 绘制聚类结果
        if save_path:
            date_save_path = save_path.replace('.png', f'_{date.strftime("%Y%m%d")}.png')
            _plot_clusters(data, clusters, date, "模式数据", date_save_path)

            # 输出聚类信息
            print(f"{date.strftime('%Y-%m-%d')} 模式数据中识别到 {len(clusters)} 个降水目标")
            for label, points in clusters.items():
                # 计算该聚类的平均降水量
                cluster_rainfall = []
                for point in points:
                    cluster_rainfall.append(rainfall[point[1], point[0]])
                avg_rainfall = np.mean(cluster_rainfall)
                max_rainfall = np.max(cluster_rainfall)

                print(
                    f"  目标 {label + 1}: 包含 {len(points)} 个格点, 平均降水量 {avg_rainfall:.1f}mm, 最大降水量 {max_rainfall:.1f}mm")

    return clusters_by_date
