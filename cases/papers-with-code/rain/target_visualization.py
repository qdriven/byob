import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import cartopy.crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib as mpl
from utils import setup_font


def identify_targets(data, threshold=25, min_size=8, separate=True):
    """
    识别降水目标区域，使用DBSCAN聚类算法

    参数:
    -----------
    data : 字典
        包含降水数据的字典
    threshold : 浮点数
        降水阈值，超过该值的区域被视为目标
    min_size : 整数
        目标的最小尺寸（网格点数）
    separate : 布尔值
        是否分别返回每个目标（True）或将它们合并为一个（False）

    返回:
    --------
    targets : 列表
        每个目标的降水数据数组列表
    """
    from sklearn.cluster import DBSCAN

    # 获取降水数据
    rainfall = data['rainfall']

    # 打印数据的基本信息
    print(f"降水数据形状: {rainfall.shape}")
    print(f"降水数据最大值: {np.max(rainfall):.1f}mm")
    print(f"降水数据最小值: {np.min(rainfall):.1f}mm")
    print(f"降水数据平均值: {np.mean(rainfall):.1f}mm")

    # 创建二值掩码：大于阈值的区域为1，其他为0
    mask = np.zeros_like(rainfall, dtype=int)
    mask[rainfall >= threshold] = 1

    # 打印超过阈值的点数
    print(f"超过阈值({threshold}mm)的点数: {np.sum(mask)}")

    # 如果没有超过阈值的点，返回空列表
    if np.sum(mask) == 0:
        print(f"没有找到超过阈值({threshold}mm)的降水格点，无法识别目标")
        return []

    # 将二维数组展平为点集合
    points = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                points.append([j, i])  # j对应经度，i对应纬度

    if not points:
        return []

    points = np.array(points)
    print(f"找到 {len(points)} 个超过阈值的点")

    # 使用固定的DBSCAN参数，与identify_and_plot_obs_clusters保持一致
    eps = 1.5
    min_samples = min_size

    print(f"使用DBSCAN参数: eps={eps}, min_samples={min_samples}")
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    # 统计聚类数量（不包括噪声点）
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_points = list(labels).count(-1)
    print(f"识别出 {n_clusters} 个目标, {noise_points} 个噪声点")

    # 如果没有找到任何聚类，返回空列表
    if n_clusters == 0:
        print("未能识别出任何目标")
        return []

    # 创建目标列表
    targets = []
    target_sizes = []

    # 为每个聚类创建一个目标数组
    for label in range(n_clusters):
        # 创建目标掩码
        target_mask = np.zeros_like(rainfall, dtype=bool)

        # 获取当前聚类的所有点
        cluster_points = points[labels == label]

        # 将聚类点标记到掩码上
        for point in cluster_points:
            j, i = point  # j对应经度，i对应纬度
            target_mask[i, j] = True

        # 创建目标数组：保留原始降水值
        target = np.zeros_like(rainfall)
        target[target_mask] = rainfall[target_mask]

        # 计算目标的平均和最大降水量
        avg_rainfall = np.mean(rainfall[target_mask])
        max_rainfall = np.max(rainfall[target_mask])

        print(f"目标 {label+1}: 包含 {len(cluster_points)} 个格点, 平均降水量 {avg_rainfall:.1f}mm, 最大降水量 {max_rainfall:.1f}mm")

        # 添加到目标列表
        targets.append(target)
        target_sizes.append(len(cluster_points))

    print(f"共识别出 {len(targets)} 个满足最小尺寸要求的目标")

    # 如果要合并所有目标为一个降水场，并且至少找到一个目标
    if not separate and targets:
        # 按大小排序目标
        sorted_targets = [target for _, target in sorted(zip(target_sizes, targets), reverse=True)]
        combined_target = np.zeros_like(rainfall)
        for target in sorted_targets:
            # 只保留非零值（不覆盖已有值）
            mask = (target > 0) & (combined_target == 0)
            combined_target[mask] = target[mask]
        print(f"已将 {len(targets)} 个目标合并为一个降水场")
        return [combined_target]

    return targets

def plot_targets_with_markers(data, targets, lon, lat, save_path=None,
                             title="降水目标标记图", date=None, threshold=25):
    """
    绘制降水等值线图，并用不同颜色的格子点标记不同的目标区域

    参数:
    -----------
    data : 字典
        包含降水数据的字典
    targets : 列表
        每个目标的降水数据数组列表
    lon : 数组
        经度数组
    lat : 数组
        纬度数组
    save_path : 字符串，可选
        保存路径
    title : 字符串，可选
        图像标题
    date : datetime对象，可选
        日期
    threshold : 浮点数，可选
        降水阈值
    """
    # 设置中文字体
    setup_font()

    # 创建图形和坐标轴
    fig = plt.figure(figsize=(12, 10), dpi=150, facecolor='white')
    ax = plt.axes(projection=ccrs.PlateCarree(), facecolor='white')

    # 设置地图范围
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # 创建绘图网格
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)

    # 获取降水数据
    rainfall = data['rainfall']

    # 设置降水等级和颜色
    levels = [0.1, 10, 25, 50, 100, 250, 500]
    colors = ['#E8F5E9', '#A5D6A7', '#4CAF50', '#81D4FA', '#2196F3', '#E040FB', '#9C27B0']
    cmap = LinearSegmentedColormap.from_list('rainfall', colors, N=len(levels))
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # 绘制填充等值图
    cf = ax.contourf(lon_mesh, lat_mesh, rainfall, levels=levels,
                     cmap=cmap, norm=norm, extend='both', alpha=0.7, zorder=5)

    # 添加地图要素
    ax.coastlines(linewidth=0.8, zorder=15)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=2)
    gl.top_labels = False
    gl.right_labels = False

    # 为每个目标绘制标记点
    marker_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'yellow', 'black']
    legend_elements = []

    for i, target in enumerate(targets):
        # 获取目标区域的掩码
        target_mask = target > threshold

        if np.sum(target_mask) == 0:
            continue

        # 获取目标区域的坐标
        y_indices, x_indices = np.where(target_mask)

        # 转换为经纬度坐标
        target_lons = lon[x_indices]
        target_lats = lat[y_indices]

        # 获取当前目标的颜色
        color = marker_colors[i % len(marker_colors)]

        # 绘制标记点
        ax.scatter(target_lons, target_lats, color=color, marker='s', s=15,
                  alpha=0.8, edgecolor='none', zorder=10, label=f'目标 {i+1}')

        # 计算目标区域的平均降水量
        avg_rainfall = np.mean(rainfall[target_mask])

        # 计算目标中心
        center_lon = np.mean(target_lons)
        center_lat = np.mean(target_lats)

        # 在目标中心添加标签
        ax.text(center_lon, center_lat, f'{i+1}: {avg_rainfall:.1f}mm',
                fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                zorder=25)

        # 添加到图例元素
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                              markerfacecolor=color, markersize=8,
                              label=f'目标 {i+1}: {np.sum(target_mask)} 格点'))

    # 添加颜色条
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=20)
    cbar.set_label('降水量 (mm)')

    # 设置标题
    if date:
        date_str = date.strftime("%Y-%m-%d")
        title_text = f"{title} - {date_str}"
    else:
        title_text = title

    ax.set_title(title_text)

    # 添加图例
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8)

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', transparent=False)
        plt.close()
    else:
        plt.show()


def plot_observation_vs_model_with_markers(obs_data, obs_targets, model_data, model_targets,
                                          date, save_path=None, threshold=25):
    """
    绘制观测数据与模式数据对比图，并用不同颜色的格子点标记不同的目标区域

    参数:
    -----------
    obs_data : 字典
        观测数据字典
    obs_targets : 列表
        观测目标列表
    model_data : 字典
        模式数据字典
    model_targets : 列表
        模式目标列表
    date : datetime对象
        日期
    save_path : 字符串，可选
        保存路径
    threshold : 浮点数，可选
        降水阈值
    """
    # 设置中文字体
    setup_font()

    # 创建图形和坐标轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10),
                                   subplot_kw={'projection': ccrs.PlateCarree(), 'facecolor': 'white'},
                                   facecolor='white')

    # 获取观测数据
    obs_lon = obs_data['lon']
    obs_lat = obs_data['lat']
    obs_rainfall = obs_data['rainfall']

    # 获取模式数据
    model_lon = model_data['lon']
    model_lat = model_data['lat']
    model_rainfall = model_data['rainfall']

    # 创建绘图网格
    obs_lon_mesh, obs_lat_mesh = np.meshgrid(obs_lon, obs_lat)
    model_lon_mesh, model_lat_mesh = np.meshgrid(model_lon, model_lat)

    # 设置降水等级和颜色
    levels = [0.1, 10, 25, 50, 100, 250, 500]
    colors = ['#E8F5E9', '#A5D6A7', '#4CAF50', '#81D4FA', '#2196F3', '#E040FB', '#9C27B0']
    cmap = LinearSegmentedColormap.from_list('rainfall', colors, N=len(levels))
    norm = mcolors.BoundaryNorm(levels, cmap.N)

    # 设置地图范围
    for ax in [ax1, ax2]:
        ax.set_extent([110, 114, 32, 36], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.8, zorder=15)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=2)
        gl.top_labels = False
        gl.right_labels = False

    # 绘制观测数据
    cf1 = ax1.contourf(obs_lon_mesh, obs_lat_mesh, obs_rainfall, levels=levels,
                      cmap=cmap, norm=norm, extend='both', alpha=0.7, zorder=5)

    # 绘制模式数据
    cf2 = ax2.contourf(model_lon_mesh, model_lat_mesh, model_rainfall, levels=levels,
                      cmap=cmap, norm=norm, extend='both', alpha=0.7, zorder=5)

    # 为观测目标绘制标记点
    marker_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'yellow', 'black']
    obs_legend_elements = []

    for i, target in enumerate(obs_targets):
        # 获取目标区域的掩码
        target_mask = target > threshold

        if np.sum(target_mask) == 0:
            continue

        # 获取目标区域的坐标
        y_indices, x_indices = np.where(target_mask)

        # 转换为经纬度坐标
        target_lons = obs_lon[x_indices]
        target_lats = obs_lat[y_indices]

        # 获取当前目标的颜色
        color = marker_colors[i % len(marker_colors)]

        # 绘制标记点
        ax1.scatter(target_lons, target_lats, color=color, marker='s', s=15,
                   alpha=0.8, edgecolor='none', zorder=10)

        # 计算目标区域的平均降水量
        avg_rainfall = np.mean(obs_rainfall[target_mask])

        # 计算目标中心
        center_lon = np.mean(target_lons)
        center_lat = np.mean(target_lats)

        # 在目标中心添加标签
        ax1.text(center_lon, center_lat, f'{i+1}: {avg_rainfall:.1f}mm',
                fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                zorder=25)

        # 添加到图例元素
        obs_legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                 markerfacecolor=color, markersize=8,
                                 label=f'观测目标 {i+1}: {np.sum(target_mask)} 格点'))

    # 为模式目标绘制标记点
    model_legend_elements = []

    for i, target in enumerate(model_targets):
        # 获取目标区域的掩码
        target_mask = target > threshold

        if np.sum(target_mask) == 0:
            continue

        # 获取目标区域的坐标
        y_indices, x_indices = np.where(target_mask)

        # 转换为经纬度坐标
        target_lons = model_lon[x_indices]
        target_lats = model_lat[y_indices]

        # 获取当前目标的颜色
        color = marker_colors[i % len(marker_colors)]

        # 绘制标记点
        ax2.scatter(target_lons, target_lats, color=color, marker='s', s=15,
                   alpha=0.8, edgecolor='none', zorder=10)

        # 计算目标区域的平均降水量
        avg_rainfall = np.mean(model_rainfall[target_mask])

        # 计算目标中心
        center_lon = np.mean(target_lons)
        center_lat = np.mean(target_lats)

        # 在目标中心添加标签
        ax2.text(center_lon, center_lat, f'{i+1}s: {avg_rainfall:.1f}mm',
                fontsize=10, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                zorder=25)

        # 添加到图例元素
        model_legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                   markerfacecolor=color, markersize=8,
                                   label=f'模式目标 {i+1}s: {np.sum(target_mask)} 格点'))

    # 添加颜色条
    cbar = fig.colorbar(cf1, ax=[ax1, ax2], orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label('降水量 (mm)')

    # 设置标题
    date_str = date.strftime("%Y-%m-%d")
    ax1.set_title(f"观测降水与目标 - {date_str}")
    ax2.set_title(f"模式降水与目标 - {date_str}")

    # 添加图例
    if obs_legend_elements:
        ax1.legend(handles=obs_legend_elements, loc='lower left', fontsize=8)
    if model_legend_elements:
        ax2.legend(handles=model_legend_elements, loc='lower left', fontsize=8)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none', transparent=False)
        plt.close()
    else:
        plt.show()


# 示例用法
if __name__ == "__main__":
    from data_process import read_observation_data, read_model_data, interpolate_to_grid

    # 设置数据路径
    obs_data_path = '观测数据202107.txt'
    model_data_folder = '模式数据/'
    output_folder = 'output/target_visualization'

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 目标日期
    target_date = datetime(2021, 7, 18)

    # 读取数据
    daily_rainfall = read_observation_data(obs_data_path)
    grid_data = interpolate_to_grid(daily_rainfall)
    model_data = read_model_data(model_data_folder)

    # 获取目标日期的数据
    obs_data = grid_data[target_date]
    model_data_day = model_data[target_date]

    # 识别目标
    obs_targets = identify_targets(obs_data, threshold=25, min_size=8, separate=True)
    model_targets = identify_targets(model_data_day, threshold=25, min_size=8, separate=True)

    # 绘制观测目标
    plot_targets_with_markers(
        obs_data, obs_targets, obs_data['lon'], obs_data['lat'],
        save_path=os.path.join(output_folder, f'观测目标标记图_{target_date.strftime("%Y%m%d")}.png'),
        title="观测降水目标标记图", date=target_date
    )

    # 绘制模式目标
    plot_targets_with_markers(
        model_data_day, model_targets, model_data_day['lon'], model_data_day['lat'],
        save_path=os.path.join(output_folder, f'模式目标标记图_{target_date.strftime("%Y%m%d")}.png'),
        title="模式降水目标标记图", date=target_date
    )

    # 绘制观测与模式对比图
    plot_observation_vs_model_with_markers(
        obs_data, obs_targets, model_data_day, model_targets, target_date,
        save_path=os.path.join(output_folder, f'观测模式目标对比图_{target_date.strftime("%Y%m%d")}.png')
    )
